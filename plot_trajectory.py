"""
    Plot the optimization path in the space spanned by principle directions.
"""

import numpy as np
import torch
import copy
import math
import h5py
import os
import argparse
import model_loader
import net_plotter
from projection_new import setup_PCA_directions, project_trajectory
import plot_2D

import torchvision.models as models
import torch.nn as nn 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot optimization trajectory')
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--model', default='resnet56', help='trained models')
    parser.add_argument('--model_folder', default='', help='folders for models to be projected')
    parser.add_argument('--dir_type', default='weights',
        help="""direction type: weights (all weights except bias and BN paras) |
                                states (include BN.running_mean/var)""")
    parser.add_argument('--ignore', default='', help='ignore bias and BN paras: biasbn (no bias or bn)')
    parser.add_argument('--prefix', default='model_', help='prefix for the checkpint model')
    parser.add_argument('--suffix', default='.t7', help='prefix for the checkpint model')
    parser.add_argument('--start_epoch', default=0, type=int, help='min index of epochs')
    parser.add_argument('--max_epoch', default=300, type=int, help='max number of epochs')
    parser.add_argument('--save_epoch', default=1, type=int, help='save models every few epochs')
    parser.add_argument('--dir_file', default='', help='load the direction file for projection')

    args = parser.parse_args()

    #--------------------------------------------------------------------------
    # load the final model
    #--------------------------------------------------------------------------
    last_model_file = args.model_folder + '/merge_{}-checkpoint.pth.tar'.format(args.max_epoch)
    # net = model_loader.load(args.dataset, args.model, last_model_file)
    net = models.resnet50()
    features_number = net.fc.in_features
    net.fc = nn.Linear(features_number, 10)

    net.load_state_dict(torch.load(last_model_file, map_location='cpu'))
    w = net_plotter.get_weights(net)
    s = net.state_dict()

    #--------------------------------------------------------------------------
    # collect models to be projected
    #--------------------------------------------------------------------------
    model_files = []
    for epoch in range(args.max_epoch):
        model_file = args.model_folder + '/merge_{}-checkpoint.pth.tar'.format(epoch)
        assert os.path.exists(model_file), 'model %s does not exist' % model_file
        model_files.append(model_file)

    #--------------------------------------------------------------------------
    # load or create projection directions
    #--------------------------------------------------------------------------
    if args.dir_file:
        dir_file = args.dir_file
    else:
        dir_file = setup_PCA_directions(args, model_files, w, s)

    print('Set PCA')
    #--------------------------------------------------------------------------
    # projection trajectory to given directions
    #--------------------------------------------------------------------------
    print('Plot trajectory')
    proj_file = project_trajectory(dir_file, w, s,
                                model_files, args.dir_type, 'cos')
    plot_2D.plot_trajectory(proj_file, dir_file)
