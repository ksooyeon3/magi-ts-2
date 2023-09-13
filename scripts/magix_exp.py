# import system package
import os
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import torch
from optparse import OptionParser

# import customize lib
import utils # experiment

# import magix
from magix.dynamic import nnModule
from magix.inference import FMAGI # inferred module

def main():
    # read in option
    usage = "usage:%prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-p", "--parameter", dest = "parameter_file_path",
                      type = "string", help = "path to the parameter file")
    parser.add_option("-r", "--result_dir", dest = "result_dir",
                      type = "string", help = "path to the result directory")
    parser.add_option("-s", dest = "random_seed", default = None,
                      type = "string", help = "random seed")
    (options, args) = parser.parse_args()

    # process parser information
    result_dir = options.result_dir
    if (not os.path.exists(result_dir)):
        os.makedirs(result_dir, exist_ok=True)
    seed = options.random_seed
    if (seed is None):
        seed = (np.datetime64('now').astype('int')*104729) % 1e9
        seed = str(int(seed))

    # read in parameters
    parameters = utils.params()
    parameters.read(options.parameter_file_path)
    if (parameters.get('experiment','seed') is None):
        parameters.add('experiment','seed',seed)

    # read in data
    example = parameters.get('data','example')
    if (example is None):
        raise ValueError('parameter data: example must be provided!')
    data = np.loadtxt('data/%s.txt' %(example))
    tdata = data[:,0] # time
    xdata = data[:,1:] # component values
    no_comp = xdata.shape[1] # number of components
    # read in number of trainning points
    no_train = parameters.get('data','no_train')
    if (no_train is None):
        no_train = int((tdata.size-1)/2) + 1
        parameters.add('data','no_train',no_train)
    no_train = int(no_train)
    obs_idx = np.linspace(0,int((tdata.size-1)/2),no_train).astype(int)
    # obtain noise parameters
    noise = parameters.get('data','noise')
    noise = [float(x) for x in noise.split(',')]
    if (len(noise) != no_comp):
        if (len(noise) == 1):
            noise = [noise[0] for i in range(no_comp)]
        else:
            raise ValueError('noise parameters must have %d components!' %(no_comp))

    # read in experiment set up
    no_run = int(parameters.get('experiment','no_run'))
    # initialize/read in random seed for the data noise
    seed = int(parameters.get('experiment','seed'))
    exp_seed = utils.seed()
    exp_seed_file = parameters.get('experiment','seed_file')
    if (exp_seed_file is None):
        exp_seed.random(no_run, seed)
    else:
        exp_seed.load(exp_seed_file)
    exp_seed_file = os.path.join(result_dir, 'exp_seed.txt')
    exp_seed.save(exp_seed_file)
    parameters.add('experiment','seed_file',exp_seed_file)

    # read in model flag and model set-up
    # magix number of iterations
    magix_no_iter = int(parameters.get('experiment','no_iteration'))
    # magix robust parameter
    magix_robust_eps = float(parameters.get('experiment','robust_eps'))
    # magix hyperparameter update
    magix_hyperparams_update = bool(int(parameters.get('experiment','hyperparams_update')))
    # magix parameters
    magix_node = parameters.get('experiment','hidden_node')
    magix_node = [no_comp] + [int(x) for x in magix_node.split(',')] + [no_comp]
    # set up heading for the output file
    magix_output_path = os.path.join(result_dir, 'magix.txt')
    magix_output = open(magix_output_path, 'w')
    magix_output.write('run,infer_time,uq_time')
    # set up heading for the output file
    magix_output_path = os.path.join(result_dir, 'magix.txt')
    magix_output = open(magix_output_path, 'w')
    magix_output.write('run,time')
    # heading for the output files
    for i in range(no_comp):
        for ptype in ['imputation','forecast','overall']:
            magix_output.write(',rmse_c%d_%s' %(i,ptype))
    magix_output.write('\n')
    magix_output.close()

    # save parameters file
    parameters.save(result_dir)

    # run experiment
    for k in range(no_run):
        # data preprocessing
        obs = []
        np.random.seed(exp_seed.get(k)) # set random seed for noise
        for i in range(no_comp):
            tobs = tdata[obs_idx].copy()
            yobs = xdata[obs_idx,i].copy() + np.random.normal(0,noise[i],no_train)
            obs.append(np.hstack((tobs.reshape(-1,1),yobs.reshape(-1,1))))

        # set random seed
        torch.manual_seed(exp_seed.get(k))
        # inference/learning
        start_time = time.time()
        fOde = nnModule(magix_node) # define nn dynamic
        magix_model = FMAGI(obs,fOde,grid_size=161,interpolation_orders=3)
        tinfer, xinfer = magix_model.map(
                max_epoch=magix_no_iter,
                learning_rate=1e-3, decay_learning_rate=True,
                robust=(magix_robust_eps>0), robust_eps=magix_robust_eps,
                hyperparams_update=magix_hyperparams_update, dynamic_standardization=True,
                verbose=False, returnX=True)
        end_time = time.time()
        x0 = xinfer[0,:].squeeze()
        irecon = np.linspace(0,tdata.size-1,321).astype(int) # reconstruction indices
        xrecon = magix_model.predict(x0,tdata[irecon])
        # performance evaluation
        run_time = end_time - start_time
        magix_output = open(magix_output_path, 'a')
        magix_output.write('%d,%s' %(k,run_time))
        for i in range(no_comp):
            xi_error = xrecon[:,i] - xdata[irecon,i]
            xi_rmse_imputation = np.sqrt(np.mean(np.square(xi_error[:161])))
            magix_output.write(',%s' %(xi_rmse_imputation))
            xi_rmse_forecast = np.sqrt(np.mean(np.square(xi_error[161:])))
            magix_output.write(',%s' %(xi_rmse_forecast))
            xi_rmse_overall = np.sqrt(np.mean(np.square(xi_error)))
            magix_output.write(',%s' %(xi_rmse_overall))
        magix_output.write('\n')
        magix_output.close()
        # release memory
        del fOde
        del magix_model


if __name__ == "__main__":
    main()
