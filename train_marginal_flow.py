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

def build_training_dataset(marginal_draws,holo_spectra,N_gwb_bins,Tspan,device,B=5):
    '''
    Helper function to build the training dataset.

    Arguments
    -----------------
    marginal_draws (array) : The masked array of parameter draws
    holo_spectra (array)   : The corresponding Holodeck GWB spectra
    N_gwb_bins (int)       : The number of frequency bins to use
    Tspan (float)          : The PTA timespan.
    B (int)                : Domain to transform the dataset to for ease of use. 
                             Should be a small-ish integer, but the specific choice shouldn't matter.

    Returns
    -----------------
    training_data (pytorch tensor) : Training dataset for the normalizing flow, linearly transformed to the domain of [-B,B].
    '''

    ## join the Holodeck spectra and the marginal parameter draws into the training dataset
    training_data = np.concatenate((marginal_draws.transpose((1,2,0)),
                                    holo_spectra.transpose((1,2,0))),
                                    axis=2)

    ## get the GWB frequencies based on N_gwb_bins and the PTA timespan
    frequencies = np.arange(1/Tspan, (N_gwb_bins+0.001)/Tspan, 1/Tspan)
    ## add (the log of) the GWB frequencies to the training dataset
    training_data[...,-N_gwb_bins:] = 0.5 * np.log10(training_data[...,-N_gwb_bins:]**2 /(12*np.pi**2 * frequencies[None, None, :]**3 * args.Tspan))

    ## transform to domain [-B,B]
    initial_min = np.min(training_data, axis = 0).min(axis = 0)
    initial_max = np.max(training_data, axis = 0).max(axis = 0)
    mean = (initial_max + initial_min)/2
    half_range = (initial_max - initial_min)/2
    training_data = B * (training_data - mean)/half_range

    ## make the pytorch tensor
    training_data = torch.tensor(training_data, device=device, dtype=torch.float32)

    ## save some information so we can invert the transformation later
    transform_info = {'B':B,
                      'half_range':half_range,
                      'mean':mean
                      }
    
    return training_data, transform_info

def postprocess_plot_compare(savedir,training_data,compare_idx=None,val_compare_idx=None,
                             flow_filename='marginal_condflow.pkl',val_filename='val_chain.pkl',
                             show=False):
    '''
    Function to load the trained flow, perform some postprocessing, and create plots/comparisons.

    Arguments
    ---------------
    savedir (str) : '/path/to/directory/with/flow/'
    training_data (array) : The training dataset.
    compare_idx (int) : (optional) Specific astro params training set index to compare on if desired. Default None (random).
    val_compare_idx (int) : (optional) Specific astro params validation set index to compare on if desired. Default None (random).
    flow_filename (str) : Name of the saved flow file. Default 'marginal_condflow.pkl'. Must be in [savedir].
    val_filename (str) : Name of the validation file. Default 'val_chain.pkl'. Must be in [savedir]
    show (bool)            : Whether to show plots. (Default False)
    
    '''
    ## load the trained flow and auxiliaries
    gwb_flow_dist, info_dict, _, _ = torch.load(savedir+'/'+flow_filename,weights_only=False)

    ## load the validation set info
    validation_set, sample_filt, realization_filt = torch.load(savedir+'/'+val_filename,weights_only=False)
    ## filter the training data to just the training set
    training_set = training_data[np.ix_(sample_filt,realization_filt)]

    ## get a random set of astro parameters
    if compare_idx is None:
        compare_idx = random.randint(0,training_set.shape[0] - 1)
    if val_compare_idx is None:
        val_compare_idx = random.randint(0,validation_set.shape[0] - 1)

    ## to undo the transform used for training
    half_range = info_dict['transform_info']['half_range']
    mean = info_dict['transform_info']['mean']
    B = info_dict['transform_info']['B']
    inverse_transform =  half_range[-info_dict['input_dim']:]/B + mean[-info_dict['input_dim']:]
    
    ## training set sample from the training library
    reference_chain = training_set[compare_idx,:,-info_dict['input_dim']:].cpu().detach().numpy() * inverse_transform

    ## generate samples from the trained flow, conditioned on the astro parameter draw
    flow_chain = gwb_flow_dist.condition(training_set[compare_idx,0,0:info_dict['context_dim']]).sample((int(1e6),)).cpu().detach().numpy() * inverse_transform

    ## check plot directory
    plotdir = savedir+'/plots/'
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    ## input_dim is the number of GWB frequency bins
    labels = [r'$\log_{{{0}}}\rho_{{{1}}}$'.format(10, _+1) for _ in range(info_dict['input_dim'])]
    ## set plot limits
    q = .01
    ll = np.quantile(flow_chain, q = q, axis = 0)
    ul = np.quantile(flow_chain, q = 1-q, axis = 0)
    ranges = list(zip(ll, ul))

    ## compare flow samples to the training dataset
    plt.figure()
    fig = None
    fig = corner.corner(flow_chain, color='mediumorchid', fig = fig, bins=20, hist_bin_factor=2, data_kwargs={'ms':3}, hist_kwargs={'density': True, 'lw':2}, 
                    contour_kwargs={'linewidths':2}, show_titles = True, desity = True, plot_datapoints = False, range = ranges, labels = labels)
    
    fig = corner.corner(reference_chain, fig = fig, color='teal', bins=20, hist_bin_factor=2, data_kwargs={'ms':3}, hist_kwargs={'density': True, 'lw':2}, 
                        contour_kwargs={'linewidths':2},  range = ranges, labels = labels,show_titles = True,
                truth_color = 'white', desity = True, plot_datapoints = False)
    
    # plt.legend(handles=lg_lines, bbox_to_anchor=(0., 1.5, 1., .0), loc=4)
    plt.legend(bbox_to_anchor=(0., 1.5, 1., .0), loc=4)
    for ext in ['.png','.pdf']:
        plt.savefig(plotdir+'/corner_training_reference'+ext)
    if show:
        plt.show()
    plt.close()
    

    ## Do the same exercise for the validation set
    
    ## validation set sample from the training library
    validation_chain = validation_set[val_compare_idx,:,-info_dict['input_dim']:].cpu().detach().numpy() * inverse_transform
    flow_chain_val = gwb_flow_dist.condition(validation_set[val_compare_idx,0,0:info_dict['context_dim']]).sample((int(1e6),)).cpu().detach().numpy() * inverse_transform
    
    ## compare flow samples to the training dataset
    plt.figure()
    fig = None
    fig = corner.corner(flow_chain_val, color='mediumorchid', fig = fig, bins=20, hist_bin_factor=2, data_kwargs={'ms':3}, hist_kwargs={'density': True, 'lw':2}, 
                    contour_kwargs={'linewidths':2}, show_titles = True, desity = True, plot_datapoints = False, range = ranges, labels = labels)
    
    fig = corner.corner(validation_chain, fig = fig, color='teal', bins=20, hist_bin_factor=2, data_kwargs={'ms':3}, hist_kwargs={'density': True, 'lw':2}, 
                        contour_kwargs={'linewidths':2},  range = ranges, labels = labels,show_titles = True,
                truth_color = 'white', desity = True, plot_datapoints = False)
    
    # plt.legend(handles=lg_lines, bbox_to_anchor=(0., 1.5, 1., .0), loc=4)
    plt.legend(bbox_to_anchor=(0., 1.5, 1., .0), loc=4)
    for ext in ['.png','.pdf']:
        plt.savefig(plotdir+'/corner_validation_reference'+ext)
    if show:
        plt.show()
    plt.close()

    ## next steps -- add code to compare to the full, non-marginalized dataset
    ## and to compare to the equivalent dataset with the marginalized values fixed
    ## also fix the legends
    
    
    return

class MarginalFlowTrainer:
    '''
    Object to house the flow training infrastructure and execute training.

    Arguments:
    ---------------------
    training_info (dict) : Dict containing all the required training information.
    device (torch.device) : The GPU/CPU to use.
    '''
    
    def __init__(self,training_info,device):

        ## store needed info
        self.info = training_info
        self.info['B'] = training_info['transform_info']['B']
        self.info['gwb_hidden_dims'] = [self.info['input_dim']*10,self.info['input_dim']*10]
        self.info['gwb_param_dims'] = [self.info['count_bins'],
                                       self.info['count_bins'],
                                       self.info['count_bins']]
        self.info['astro_count_bins'] = 8
        self.info['astro_param_dims'] = [8,8,8]

        ## construct the neural net
        self.construct_flow(device)
        
        return

    def construct_flow(self,device):
        '''
        Function to build the flow neural network, based on the values provided in self.info.
        In principle, similar functions for different architectures can be incorporated in future if desired.
        '''
        ## build infrastructure for the context (astro) parameters
        self.astro_dist_base = dist.Uniform(torch.ones(self.info['context_dim'], device=device) * (-self.info['B']-1), 
                                            torch.ones(self.info['context_dim'], device=device) * (self.info['B']+1))

        self.astro_hypernet = AutoRegressiveNN(input_dim = self.info['context_dim'], 
                                               hidden_dims = [self.info['context_dim']*25,
                                                              self.info['context_dim']*25], 
                                               param_dims=self.info['astro_param_dims']).cuda()
        self.astro_transform = T.SplineAutoregressive(self.info['context_dim'],
                                                      self.astro_hypernet,
                                                      count_bins=self.info['astro_count_bins'], 
                                                      order = 'quadratic',
                                                      bound = self.info['B']+1).cuda()
        self.astro_flow_dist = dist.TransformedDistribution(self.astro_dist_base,[self.astro_transform])

        ## build flow infrastructure for the GWB PSD, conditioned on the marginal parameter space
        self.gwb_dist_base = dist.Uniform(torch.ones(self.info['input_dim'], device = device)*(-self.info['B']-1), 
                                          torch.ones(self.info['input_dim'], device = device)*self.info['B']+1)
        self.gwb_hypernet = ConditionalAutoRegressiveNN(self.info['input_dim'], 
                                                        self.info['context_dim'],
                                                        self.info['gwb_hidden_dims'],
                                                        param_dims=self.info['gwb_param_dims']).cuda()
        self.gwb_transform = T.ConditionalSplineAutoregressive(self.info['input_dim'],
                                                               self.gwb_hypernet,
                                                               count_bins=self.info['count_bins'],
                                                               order='quadratic',
                                                               bound = self.info['B']+1).cuda()
        self.gwb_flow_dist = dist.ConditionalTransformedDistribution(self.gwb_dist_base,[self.gwb_transform])

        return

    def train_flow(self,training_data,device,savedir,
                   validation_percent=10,steps=50,batch_size=int(1e3)):
        '''
        Function to train the marginal flow.

        Arguments
        -----------------
        training_data (torch tensor) : The training dataset
        device (torch.device)        : The GPU/CPU to train on.
        savedir (str)               : '/path/to/save/directory/'
        validation_percent (float)   : What percent of the training dataset to reserve for validation.
        
        '''
        ## first, define the validation set
        validation_sample_idxs = random.sample(range(training_data.shape[0]),
                                               k=int(training_data.shape[0]*validation_percent/100))
        validation_realization_idxs = random.sample(range(training_data.shape[1]),
                                                    k=int(training_data.shape[1]*validation_percent/100))
        validation_set = training_data[np.ix_(validation_sample_idxs,validation_realization_idxs)]
        sample_filt = np.ones(training_data.shape[0],dtype=bool)
        sample_filt[validation_sample_idxs] = False
        realization_filt = np.ones(training_data.shape[1],dtype=bool)
        realization_filt[validation_realization_idxs] = False
        training_set = training_data[np.ix_(sample_filt,realization_filt)]

        print("Saving validation set ({}% of full dataset) to {}...".format(validation_percent,savedir+'/val_chain.pkl'))
        torch.save([validation_set,sample_filt,realization_filt],savedir+'/val_chain.pkl')

        ## preliminaries
        modules = torch.nn.ModuleList([self.gwb_transform])
        optimizer = torch.optim.Adam(modules.parameters(), lr=3e-3)

        ## for the loss histograms
        count = 0
        fail_counter = 0
        self.loss_hist = []
        self.validation_loss_hist = []
        
        ## size info
        total_sample_size = training_set.shape[0]
        total_realization_size = training_set.shape[1]

        ## make sure the batch size isn't larger than one of the axes
        if batch_size > total_sample_size or batch_size > total_realization_size:
            print("Warning: requested batch size ({}) exceeds an axis of the training data.".format(batch_size))
            batch_size = int(np.min((total_sample_size,total_realization_size))/2)
            print("Setting batch_size={} (50% of the smallest dimension of the training data)".format(batch_size))
        
        ## train!
        for step in tqdm(range(steps)):
            # try:
            ## get N=batch_size random samples of the astro params and corresponding GWB spectra
            random_sample = random.sample(range(total_sample_size), k=batch_size)
            random_realization = random.sample(range(total_realization_size), k=batch_size)

            optimizer.zero_grad()

            astro_sample = training_set[random_sample,0,0:self.info['context_dim']]
            gwb_sample = training_set[random_sample,random_realization,self.info['context_dim']:]

            ## get the conditional (marginal!) log probability
            ## i.e., integral p(GWB | theta_interest, theta_nuisance) dtheta_nuisance
            ln_prob_gwb_given_astro = self.gwb_flow_dist.condition(astro_sample).log_prob(gwb_sample)

            ## compute loss
            loss = -(ln_prob_gwb_given_astro).mean()

            ## every 10 steps, check the validation loss
            if step % 10 == 0:

                ln_prob_gwb_given_astro_val = self.gwb_flow_dist.condition(validation_set[:,:,0:self.info['context_dim']]).log_prob(validation_set[:,:,self.info['context_dim']:])
                validation_loss = -(ln_prob_gwb_given_astro_val).mean()

                ## back-propagate
                loss.backward()
                validation_loss.backward()
                optimizer.step()

                ## add to histograms
                self.validation_loss_hist.append(validation_loss.item())
                self.loss_hist.append(loss.item())
                count += 1
            else:
                ## otherwise just back-propagate the loss and continue
                loss.backward()
                optimizer.step()
            # except:
            #     ## handle breaking cases
            #     print("Failed!")
            #     fail_counter += 1
            #     self.gwb_flow_dist.clear_cache()
            #     continue
        ## end of training block

        ## save the flow
        torch.save([self.gwb_flow_dist,self.info,self.loss_hist,self.validation_loss_hist],
                    savedir+'/marginal_condflow.pkl')
                                                                                                                        
        
        return

    def plot_loss(self,savedir,show=False):
        '''
        Plot the training and validation loss.

        Arguments
        ----------------
        savedir (str) : "/path/to/save/dir/'
        '''

        ## set font
        plt.rcParams['font.family'] = 'STIXGeneral'  # Closely matches Computer Modern
        plt.rcParams['mathtext.fontset'] = 'stix'    # Use STIX for math
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['legend.fontsize'] = 14

        
        iters = 10*np.arange(len(self.loss_hist))
        plt.figure()
        plt.plot(iters,self.loss_hist,c='mediumorchid',label='Training Loss')
        plt.plot(iters,self.validation_loss_hist,c='teal',label='Validation Loss')
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend(loc='upper right')
        for ext in ['.png','.pdf']:
            plt.savefig(savedir+'/loss'+ext,dpi=300)
        print("Saved loss plot to {}.".format(savedir+'/loss.png'))
        if show:
            plt.show()
        else:
            plt.close()
        
        return

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Train a marginal flow.')
    parser.add_argument('holodeck_population', type=str, help='/path/to/holodeck/population/data/file.hdf5')
    parser.add_argument('-s','--savedir', type=str, help='/path/to/save/dir/',default=None)
    parser.add_argument('--N_gwb_bins', type=int, help='Number of GWB frequency bins to use (max 30).', default=5)
    parser.add_argument('-T','--Tspan',type=float, help='Timespan of the simulated PTA dataset', default=505861299.1401643)
    parser.add_argument('--steps', type=int, help="Number of iterations to use for flow training.", default=50050)
    
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
    
    if args.savedir is None:
        savedir = './'
    else:
        if not os.path.exists(args.savedir):
            os.mkdir(args.savedir)
        savedir = args.savedir

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

    ## create the training dataset
    ## this is a pytorch tensor combining the draws and spectra, linearly transformed to a limited numerical domain
    training_data, transform_info = build_training_dataset(marginal_draws,holo_spectra,args.N_gwb_bins,args.Tspan,device)
    
    ## get the number of training parameters
    N_parameters = len(marginal_parameters)

    ## info we need to train the flow
    training_info = {'input_dim':args.N_gwb_bins,
                     'context_dim':N_parameters,
                     'count_bins':16,
                     'transform_info':transform_info}

    ## build the neural net
    flow_trainer = MarginalFlowTrainer(training_info,device)

    ## As Nima would say, let it flow!
    flow_trainer.train_flow(training_data,device,savedir,steps=args.steps)

    ## make plots
    plotdir = savedir+'/plots/'
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)
    flow_trainer.plot_loss(plotdir)


    ## postprocess
    postprocess_plot_compare(savedir,training_data)











