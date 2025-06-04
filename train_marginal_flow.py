#!/usr/bin/env python

import argparse
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import AutoRegressiveNN
from pyro.nn import ConditionalAutoRegressiveNN
import numpy as np
import torch, h5py, random, corner, pickle
from tqdm import tqdm
import copy, os
import matplotlib.pyplot as plt


def load_holodeck_population(popfile,N_gwb_bins,floor=1e-20):
    '''
    Helper function to load the holodeck population file.

    Arguments
    -----------------
    popfile (str) : '/path/to/population/data/file.hdf5'
    N_gwb_bins (int) : Number of GWB frequency bins to use.
    floor (float) : (Optional) Lower bound of the numerical prior. Regions of the parameter space with a Holodeck-produced GWB spectrum with an amplitude below this floor will be set to the floor value. Default 1e-20.

    Returns
    ---------------
    holo_draws (array) : numpy array with the parameter draws and realizations.
    holo_spectra (array) : numpy array containing the corresponding GWB spectra
    holo_info (dict) : Dictionary storing the number of parameters, samples, and realizations, alongside parameter names.
    
    '''

    ## load an preprocess the population data
    with h5py.File(popfile, 'r') as data:
        ## get the GWB samples and filter to the desired frequency bins
        holo_spectra = data['gwb'][()][:, 0:N_gwb_bins, :].transpose(1,0,2)
        print(holo_spectra.shape)
        ## set the spectrum floor
        floor_idx = np.where(holo_spectra < floor)
        holo_spectra[floor_idx] = 1e-20

        ## get the parameters varied for the dataset
        param_names = data.attrs['param_names'].astype(str)
        print("Loaded dataset with varied parameters: {}".format(param_names))
        ## get the Holodeck spectrum samples by parameter
        holo_draws = data['sample_params'][()]

    ## useful dimensions
    ## definitions:
    ## 1. parameters: the intrinsic population parameters, e.g. \theta
    ## 2. samples: samples from the population parameters \theta
    ## 3. realizations: given a specific sample \theta, draw Poisson realizations on the number of binaries
    ## 4. draws: combination across samples x realizations
    N_holo_parameters = holo_draws.shape[1]
    N_holo_realizatons = holo_spectra.shape[-1]
    N_holo_samples = holo_spectra.shape[-2]
    ## recast to shape parameters x samples x realizations
    holo_draws = np.broadcast_to(holo_draws,(N_holo_realizatons, N_holo_samples, N_holo_parameters)).transpose((2,1,0))
    ## dict with dimensions to help keep track
    holo_info = {"N_holo_samples":N_holo_samples,
                 "N_holo_parameters":N_holo_parameters,
                 "N_holo_realizatons":N_holo_realizatons,
                 "parameters":param_names}
    
    ## catch some potential problems
    assert np.all(holo_draws[:, :, 0] == holo_draws[:, :, 1]) ## sampled parameters are identical across realizations
    assert N_holo_parameters == len(param_names)

    return holo_draws, holo_spectra, holo_info

def mask_holodeck_dataset(holo_draws,initial_parameters,nuisance_parameters):
    '''
    Helper function to create a reduced version of a Holodeck dataset with labels removed for to-be-marginalized-over parameters.

    Arguments
    ------------
    holo_draws (array) : Array of parameter draws of shape len(initial_parameters) x N_samples x N_realizations
    initial_parameters (list of str) : List of parameter names sampled over in the Holodeck dataset.
    nuisance_parameters (list of str) : List of parameter names to marginalize over.
    
    Returns
    ------------
    marginal_draws (array) : Array of parameter draws in the new, marginal space.
    marginal_parameters (list of str) : List of parameter names in the new, marginal space.
    '''

    ## get indices of desired marginal parameter space
    marginal_filt = [i for i in range(len(initial_parameters)) if initial_parameters[i] not in nuisance_parameters]
    marginal_parameters = initial_parameters[marginal_filt]
    print("Marginalizing over parameters: {}".format(nuisance_parameters))
    print("Resulting marginal parameter space: {}".format(marginal_parameters))

    ## filter parameter draws to the correct indices
    marginal_draws = holo_draws[marginal_filt,:,:]

    return marginal_draws, marginal_parameters
    

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train a marginal flow.')
    parser.add_argument('holodeck_population', type=str, help='/path/to/holodeck/population/data/file.hdf5')
    parser.add_argument('--outdir', type=str, help='/path/to/save/dir/', default=None)
    parser.add_argument('--N_gwb_bins', type=int, help='Number of GWB frequency bins to use (max 30).', default=5)

    ## arguments to specify which parameters are marginalized over
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-m','--marginalize', nargs='+',type=str,
                        help='Parameter names to marginalize over. Pass as -m name_1 name_2 name_3. Only one of --marginalize, --marginalize_idx, or --marginalize_all can be specified.',
                       default=None)
    group.add_argument('-mi','--marginalize_idx', nargs='+',type=int,
                        help='Parameter indices (i.e., column indices of the holodeck dataset) to marginalize over. Pass as -m i j k. Only one of --marginalize, --marginalize_idx, or --marginalize_all can be specified.',
                       default=None)
    group.add_argument('-me','--marginalize_ext',action='store_true',help='Special case to marginalize the "ext" in phenom_ext, reducing to just the phenom parameter space')
    group.add_argument('-ma','--marginalize_all',action='store_true',help='Marginalize over all parameters, therefore recovering the full marginal prior on the GWB spectrum.')
    
    parser.add_argument('--opt_arg', type=float, help='Optional arg 1', default=10)

    
    args = parser.parse_args()

    if args.N_gwb_bins > 30:
        raise ValueError("Maximum number of frequency bins is 30 (selected: {}).".format(args.N_gwb_bins))
    
    if args.outdir is None:
        outdir = '.'
    else:
        if not os.path.exists(args.outdir):
            os.mkdir(args.outdir)
        outdir = args.outdir

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available and being used")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead")

    ## load the Holodeck population to train on
    holo_draws, holo_spectra, holo_info = load_holodeck_population(args.holodeck_population,args.N_gwb_bins)

    ## mask Holodeck dataset as appropriate for the desired marginalization
    if args.marginalize_all:
        nuisance_parameters = holo_info['parameters']
    elif args.marginalize_ext:
        nuisance_parameters = ['gsmf_alpha0', 'gpf_zbeta', 'gpf_qgamma', 'gmt_norm', 'gmt_zbeta', 'mmb_plaw']
    elif args.marginalize_idx is not None:
        nuisance_parameters = holo_info['parameters'][args.marginalize_idx]
    elif args.marginalize is not None:
        nuisance_parameters = args.marginalize
    else:
        raise RuntimeError("No marginalization was specified! Either --marginalize, --marginalize_idx, --marginalize_all, or --marginalize_ext must be provided.")

    ## mask the parameter draws (i.e., the training labels) to just the marginal parameter space
    marginal_draws, marginal_parameters = mask_holodeck_dataset(holo_draws,holo_info['parameters'],nuisance_parameters)

    ## join the Holodeck spectra and the marginal parameter draws into the training dataset
    training_data = np.concatenate((marginal_draws.transpose((1,2,0)),
                                    holo_spectra.transpose((1,2,0))),
                                    axis=2)
    print(training_data.shape)
    
    ## left off at "Some things that we need to know about the library..." in the notebook.
    ## need to add Tspan as a variable to the argparse, take log of frequencies, make the pytorch tensor object, etc.
    ## should probably package the above together into another helper function, create_training_dataset() or similar













