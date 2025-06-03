#!/usr/bin/env python

import argparse
# import pyro.distributions as dist
# import pyro.distributions.transforms as T
# from pyro.nn import AutoRegressiveNN
# from pyro.nn import ConditionalAutoRegressiveNN
# import numpy as np
# import torch, h5py, random, corner, pickle
# from tqdm import tqdm
# import copy, os
# import matplotlib.pyplot as plt


def load_holodeck_population(popfile,N_gwb_bins,floor=1e-20):
    '''
    Helper function to load the holodeck population file.

    Arguments
    -----------------
    popfile (str) : '/path/to/population/data/file.hdf5'
    N_gwb_bins (int) : Number of GWB frequency bins to use.

    floor (float) : Lower bound of the numerical prior. Regions of the parameter space with a Holodeck-produced GWB spectrum with an amplitude below this floor will be set to the floor value. Default 1e-20.
    
    '''










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
    group.add_argument('-ma','--marginalize_all',action='store_true')
    
    parser.add_argument('--opt_arg', type=float, help='Optional arg 1', default=10)

    
    args = parser.parse_args()

    print(args.marginalize)
    print(args.marginalize_idx)
    print(args.marginalize_all)
    raise ValueError('done')
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