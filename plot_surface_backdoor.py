
from dataset.poisoned_cifar10 import PoisonedCIFAR10
from dataset.poisoned_cifar100 import PoisonedCIFAR100
from dataset.poisoned_rimagenet import RestrictedImageNet
from dataset.clean_label_cifar10 import CleanLabelPoisonedCIFAR10


from pruner import *
from models.resnets import resnet20s
# ResNet18
from models.model_zoo import *
from models.densenet import *
from models.vgg import *
from models.adv_resnet import resnet20s as robust_res20s
from torch.utils.data import DataLoader, Subset
import torchvision.models as models 

import argparse
import copy
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
import dataloader
import evaluation
import projection as proj
import net_plotter
import plot_2D
import plot_1D
import model_loader
import scheduler
import mpi4pytorch as mpi
from torch.utils.data import DataLoader


def name_surface_file(args, dir_file):
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))

    # dataloder parameters
    if args.raw_data: # without data normalization
        surf_file += '_rawdata'
    if args.data_split > 1:
        surf_file += '_datasplit=' + str(args.data_split) + '_splitidx=' + str(args.split_idx)

    return surf_file + ".h5"


def setup_surface_file(args, surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already set up" % surf_file)
            return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=args.xnum)
    f['xcoordinates'] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=args.ynum)
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file


def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, comm, rank, args):
    """
        Calculate the loss values and accuracies of modified models in parallel
        using MPI reduce.
    """

    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)

    print('Computing %d values for rank %d'% (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    criterion = nn.CrossEntropyLoss()
    if args.loss_name == 'mse':
        criterion = nn.MSELoss()

    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':
            net_plotter.set_weights(net.module if args.ngpu > 1 else net, w, d, coord)
        elif args.dir_type == 'states':
            net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord)

        # Record the time to compute the loss value
        loss_start = time.time()
        loss, acc = evaluation.eval_loss(net, criterion, dataloader, args.cuda)
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        losses     = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)
        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f[loss_key][:] = losses
            f[acc_key][:] = accuracies
            f.flush()

        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc, loss_compute_time, syc_time))

    # This is only needed to make MPI run smoothly. If this process has less work than
    # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    for i in range(max(inds_nums) - len(inds)):
        losses = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)

    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))

    f.close()

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')

    ##################################### Backdoor #################################################
    parser.add_argument("--poison_ratio", type=float, default=0.01)
    parser.add_argument("--patch_size", type=int, default=5, help="Size of the patch")
    parser.add_argument("--random_loc", dest="random_loc", action="store_true", help="Is the location of the trigger randomly selected or not?")
    parser.add_argument("--upper_right", dest="upper_right", action="store_true")
    parser.add_argument("--bottom_left", dest="bottom_left", action="store_true")
    parser.add_argument("--target", default=0, type=int, help="The target class")
    parser.add_argument("--black_trigger", action="store_true")
    parser.add_argument("--clean_label_attack", action="store_true")
    parser.add_argument('--robust_model', type=str, default=None, help='checkpoint file')
    parser.add_argument("--seed", default=0, type=int, help="Random Seed")

    parser.add_argument('--arch', type=str, default='resnet18', help='network architecture')

    # data parameters
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet')
    parser.add_argument('--data', default='cifar10/data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='no data preprocessing')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')
    parser.add_argument('--workers', type=int, default=2, help='number of workers in dataloader')

    # model parameters
    parser.add_argument('--model', default='resnet56', help='model name')
    parser.add_argument('--model_folder', default='', help='the common folder that contains model_file and model_file2')
    parser.add_argument('--model_file', default='', help='path to the trained model file')
    parser.add_argument('--model_file2', default='', help='use (model_file2 - model_file) as the xdirection')
    parser.add_argument('--model_file3', default='', help='use (model_file3 - model_file) as the ydirection')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

    # direction parameters
    parser.add_argument('--dir_file', default='', help='specify the name of direction file, or the path to an eisting direction file')
    parser.add_argument('--dir_type', default='weights', help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default=None, help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--same_dir', action='store_true', default=False, help='use the same random direction for both x-axis and y-axis')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
    parser.add_argument('--surf_file', default='', help='customize the name of surface file, could be an existing file.')

    # plot parameters
    parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    #--------------------------------------------------------------------------
    # Environment setup
    #--------------------------------------------------------------------------
    if args.mpi:
        comm = mpi.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception('User selected cuda option, but cuda is not available on this machine')
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print('Rank %d use GPU %d of %d GPUs on %s' %
                (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

    #--------------------------------------------------------------------------
    # Check plotting resolution
    #--------------------------------------------------------------------------
    try:
        args.xmin, args.xmax, args.xnum = [int(a) for a in args.x.split(':')]
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [int(a) for a in args.y.split(':')]
            assert args.ymin and args.ymax and args.ynum, \
            'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

    #--------------------------------------------------------------------------
    # Load models and extract parameters
    #--------------------------------------------------------------------------
    
    # net = resnet20s()
    # print('loading tickets from {}'.format(args.ticket_dir))
    # ticket_checkpoint = torch.load(args.ticket_dir, map_location='cpu')
    # mask_checkpoint = extract_mask(ticket_checkpoint['state_dict'])
    # if len(mask_checkpoint):
    #     prune_model_custom(net, mask_checkpoint)
    # net.load_state_dict(ticket_checkpoint['state_dict'])
    # check_sparsity(net)

    # prepare dataset
    if args.dataset == 'cifar10':
        print('Dataset = CIFAR10')
        classes = 10
        if args.clean_label_attack:
            print('Clean Label Attack')
            robust_model = robust_res20s(num_classes = classes)
            robust_weight = torch.load(args.robust_model, map_location='cpu')
            if 'state_dict' in robust_weight.keys():
                robust_weight = robust_weight['state_dict']
            robust_model.load_state_dict(robust_weight)
            train_set = CleanLabelPoisonedCIFAR10(args.data, poison_ratio=args.poison_ratio, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, black_trigger=args.black_trigger, robust_model=robust_model)
        else:
            train_set = PoisonedCIFAR10(args.data, train=True, poison_ratio=args.poison_ratio, patch_size=args.patch_size,
                                        random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                        target=args.target, black_trigger=args.black_trigger)

        sub_train_set = Subset(train_set, list(range(50000))[:1000])

        clean_testset = PoisonedCIFAR10(args.data, train=False, poison_ratio=0, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, black_trigger=args.black_trigger)
        poison_testset = PoisonedCIFAR10(args.data, train=False, poison_ratio=1, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, black_trigger=args.black_trigger)
        train_dl = DataLoader(sub_train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        clean_test_dl = DataLoader(clean_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        poison_test_dl = DataLoader(poison_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    elif args.dataset == 'cifar100':
        print('Dataset = CIFAR100')
        classes = 100
        train_set = PoisonedCIFAR100(args.data, train=True, poison_ratio=args.poison_ratio, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, black_trigger=args.black_trigger)
        clean_testset = PoisonedCIFAR100(args.data, train=False, poison_ratio=0, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, black_trigger=args.black_trigger)
        poison_testset = PoisonedCIFAR100(args.data, train=False, poison_ratio=1, patch_size=args.patch_size,
                                    random_loc=args.random_loc, upper_right=args.upper_right, bottom_left=args.bottom_left, 
                                    target=args.target, black_trigger=args.black_trigger)
        
        sub_train_set = Subset(train_set, list(range(50000))[:1000])

        train_dl = DataLoader(sub_train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        clean_test_dl = DataLoader(clean_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        poison_test_dl = DataLoader(poison_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    elif args.dataset == 'rimagenet':
        print('Dataset = Restricted ImageNet')
        classes = 9 
        dataset = RestrictedImageNet(args.data)
        train_dl, _, _ = dataset.make_loaders(workers=args.workers, batch_size=args.batch_size, poison_ratio=args.poison_ratio, target=args.target, patch_size=args.patch_size, black_trigger=args.black_trigger, subset=1000)
        _, clean_test_dl = dataset.make_loaders(only_val=True, workers=args.workers, batch_size=args.batch_size, poison_ratio=0, target=args.target, patch_size=args.patch_size, black_trigger=args.black_trigger)
        _, poison_test_dl = dataset.make_loaders(only_val=True, workers=args.workers, batch_size=args.batch_size, poison_ratio=1, target=args.target, patch_size=args.patch_size, black_trigger=args.black_trigger)
    else:
        raise ValueError('Unknow Datasets')

    # prepare model
    if args.dataset == 'rimagenet':
        if args.arch == 'resnet18':
            net = models.resnet18(num_classes=classes)
        else:
            raise ValueError('Unknow architecture')
    else:
        if args.arch == 'resnet18':
            net = ResNet18(num_classes=classes)
        elif args.arch == 'resnet20':
            net = resnet20s(num_classes=classes)
        elif args.arch == 'densenet100':
            net = densenet_100_12(num_classes=classes)
        elif args.arch == 'vgg16':
            net = vgg16_bn(num_classes=classes)
        else:
            raise ValueError('Unknow architecture')


    print('===> loading weight from {} <==='.format(args.model_file))
    pretrained_weight = torch.load(args.model_file, map_location='cpu')

    if 'state_dict' in pretrained_weight:
        pretrained_weight = pretrained_weight['state_dict']
    sparse_mask = extract_mask(pretrained_weight)
    if len(sparse_mask) > 0:
        prune_model_custom(net, sparse_mask)
    net.load_state_dict(pretrained_weight)
    check_sparsity(net)

    # net = model_loader.load(args.dataset, args.model, args.model_file)
    w = net_plotter.get_weights(net) # initial parameters
    s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references

    # use single gpu
    # if args.ngpu > 1:
    #     # data parallel with multiple GPUs on a single node
    #     net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    #--------------------------------------------------------------------------
    # Setup the direction file and the surface file
    #--------------------------------------------------------------------------
    dir_file = net_plotter.name_direction_file(args) # name the direction file
    if rank == 0:
        net_plotter.setup_direction(args, dir_file, net)

    surf_file = name_surface_file(args, dir_file)
    if rank == 0:
        setup_surface_file(args, surf_file, dir_file)

    # wait until master has setup the direction file and surface file
    mpi.barrier(comm)

    # load directions
    d = net_plotter.load_directions(dir_file)
    # calculate the consine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    #--------------------------------------------------------------------------
    # Setup dataloader
    #--------------------------------------------------------------------------
    # download CIFAR10 if it does not exit
    # if rank == 0 and args.dataset == 'cifar10':
    #     torchvision.datasets.CIFAR10(root=args.dataset + '/data', train=True, download=True)

    mpi.barrier(comm)

    # trainloader, testloader = dataloader.load_dataset(args.dataset, args.datapath,
    #                             args.batch_size, args.threads, args.raw_data,
    #                             args.data_split, args.split_idx,
    #                             args.trainloader, args.testloader)


    # freq = True if args.freq else False
    # random_loc = True if args.random_loc else False
    # train_set = PoisonedCIFAR10(args.data, train=True, poison_ratio=args.poison_ratio, patch_size=args.patch_size, freq=freq, random_loc=random_loc, target=args.target)
    # clean_testset = PoisonedCIFAR10(args.data, train=False, poison_ratio=0, patch_size=args.patch_size, freq=freq, random_loc=random_loc, target=args.target)
    # poison_testset = PoisonedCIFAR10(args.data, train=False, poison_ratio=1.0, patch_size=args.patch_size, freq=freq, random_loc=random_loc, target=args.target)

    # train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # clean_test_dl = DataLoader(clean_testset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # poison_test_dl = DataLoader(poison_testset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    #--------------------------------------------------------------------------
    # Start the computation
    #--------------------------------------------------------------------------
    crunch(surf_file, net, w, s, d, train_dl, 'train_loss', 'train_acc', comm, rank, args)
    # crunch(surf_file, net, w, s, d, poison_test_dl, 'test_loss', 'test_acc', comm, rank, args)

    #--------------------------------------------------------------------------
    # Plot figures
    #--------------------------------------------------------------------------
    if args.plot and rank == 0:
        plot_2D.plot_2d_contour(surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, args.show)
        # if args.y and args.proj_file:
        #     plot_2D.plot_contour_trajectory(surf_file, dir_file, args.proj_file, 'train_loss', args.show)
        # elif args.y:
        #     plot_2D.plot_2d_contour(surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, args.show)
        # else:
        #     plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show)
