from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from dataprocess import AEDPatchDataset, SequentialAEDPatchSampler, SequentialShapeRandomAEDPatchSampler
from aednet import ResAEDNet
from shutil import copyfile
import math

def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--indir', type=str, default='../data/AEDNetTestset_BA', help='input folder (events)')
    parser.add_argument('--outdir', type=str, default='./results', help='output folder (estimated events properties)')
    parser.add_argument('--dataset', type=str, default='testset.txt', help='shape set file name')
    parser.add_argument('--modeldir', type=str, default='../models/BA_noise_removal_model', help='model folder')
    parser.add_argument('--model', type=str, default='AEDNet', help='names of trained models, can evaluate multiple models')
    parser.add_argument('--modelpostfix', type=str, default='_model.pth', help='model file postfix')
    parser.add_argument('--parmpostfix', type=str, default='_params.pth', help='parameter file postfix')
    parser.add_argument('--n_neighbours', type=int, default=200, help='nearest neighbour used for inflation step')
    parser.add_argument('--sampling', type=str, default='full', help='sampling strategy, any of:\n'
                        'full: evaluate all points in the dataset\n'
                        'sequential_shapes_random_patches: pick n random points from each shape as patch centers, shape order is not randomized')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)')
    parser.add_argument('--label', type=int, default=True, help='whether test data accompanies with label')
    parser.add_argument('--seed', type=int, default=40938661, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=0, help='batch size, if 0 the training batch size is used')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--nrun', type=int, default=1, help='nrun')
    parser.add_argument('--shapenum', type=int, default=100000, help='how many points in a shape')
    parser.add_argument('--shapename', type=str, default='MAH00444_50{i}', help='shape to evaluate format : name{i}')
    parser.add_argument('--x_frame', type=int,
                        default=1280, help='the width of the frame')
    parser.add_argument('--y_frame', type=int,
                        default=720, help='the height of the frame')
    parser.add_argument('--x_lim', type=float,
                        default=25, help='the limitation of x and y compare to the whole frame')
    parser.add_argument('--y_lim', type=float,
                        default=15, help='the limitation of x and y compare to the whole frame')


    return parser.parse_args()

def init_res_directory(opt):
    # copy input shape to results directory at first iteration
    if opt.nrun == 1:
        src = os.path.join(opt.indir, opt.shapename.rsplit("{", 1)[0] + ".npy")
        dst = os.path.join(opt.outdir, opt.shapename.format(i = 0) + ".npy" )
        copyfile(src, dst)


def eval_pcpnet(opt):
    # get a list of model names
    model_name = opt.model
    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    model_filename = os.path.join(opt.modeldir, opt.model+opt.modelpostfix)
    param_filename = os.path.join(opt.modeldir, opt.model+opt.parmpostfix)

    # load model and training parameters
    trainopt = torch.load(param_filename)
    if opt.batchSize == 0:
        model_batchSize = trainopt.batchSize
    else:
        model_batchSize = opt.batchSize
    # get indices in targets and predictions corresponding to each output

    if opt.x_frame == 0:
        x_frame = trainopt.x_frame
    else:
        x_frame = opt.x_frame

    if opt.y_frame == 0:
        y_frame = trainopt.y_frame
    else:
        y_frame = opt.y_frame

    if opt.x_lim == 0:
        x_lim = trainopt.x_lim
    else:
        x_lim = opt.x_lim

    if opt.y_lim == 0:
        y_lim = trainopt.y_lim
    else:
        y_lim = opt.y_lim

    pred_dim = 2
    dataset = AEDPatchDataset(
        root=opt.outdir, shapes_list_file=opt.dataset,
        shape_num=opt.shapenum,
        points_per_patch=trainopt.points_per_patch,
        x_limitation=x_lim,
        y_limitation=y_lim,
        x_frame=x_frame,
        y_frame=y_frame,
        seed=opt.seed,
        center=trainopt.patch_center,
        cache_capacity=opt.cache_capacity,
        shape_names = [opt.shapename.format(i = opt.nrun-1)],
        train=False,
        label=opt.label)

    if opt.sampling == 'full':
        datasampler = SequentialAEDPatchSampler(dataset)
    elif opt.sampling == 'sequential_shapes_random_patches':
        datasampler = SequentialShapeRandomAEDPatchSampler(
            dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            sequential_shapes=True,
            identical_epochs=False)
    else:
        raise ValueError('Unknown sampling strategy: %s' % opt.sampling)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=datasampler,
        batch_size=model_batchSize,
        num_workers=int(opt.workers))


    regressor = ResAEDNet(
        num_points=trainopt.points_per_patch,
        output_dim=pred_dim,
        use_point_stn=trainopt.use_point_stn,
        use_feat_stn=trainopt.use_feat_stn,
        sym_op=trainopt.sym_op)

    regressor.load_state_dict(torch.load(model_filename))
    regressor.cuda()

    shape_ind = 0
    shape_patch_offset = 0
    if opt.sampling == 'full':
        n = len(dataset.shape_patch_count)
        shape_patch_count = (n-1)*opt.shapenum+dataset.shape_patch_count[n-1]
    elif opt.sampling == 'sequential_shapes_random_patches':
        shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
    else:
        raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
    shape_properties = torch.FloatTensor(shape_patch_count, 3).zero_()
    shape_pol = torch.FloatTensor(shape_patch_count).zero_()
    if opt.label:
        shape_wrong = torch.FloatTensor(shape_patch_count, 4).zero_()
        shape_label = torch.FloatTensor(shape_patch_count).zero_()

    # append model name to output directory and create directory if necessary
    model_outdir = os.path.join(opt.outdir, model_name)
    if not os.path.exists(model_outdir):
        os.makedirs(model_outdir)

    num_batch = len(dataloader)
    batch_enum = enumerate(dataloader, 0)
    if opt.label:
        events = 0
        noise = 0
        k = 0
    num = 0


    regressor.eval()
    for batchind, data in batch_enum:

        # get batch, convert to variables and upload to GPU
        if opt.label:
            points, labels, originals, pol = data
            labels = labels.cuda()
        else:
            points, originals, pol = data

        points = Variable(points, volatile=True)
        points = points.transpose(2, 1)
        points = points.cuda()
        originals = originals.cuda()
        pol = pol.cuda()

        pred, _, _, _ = regressor(points)
        pred = pred.data
        _, pred = torch.max(F.softmax(pred, dim=1), 1)

        print('[%s %d/%d] shape %s' % (model_name, batchind, num_batch - 1, dataset.shape_names[shape_ind]))
        batch_offset = 0
        j = 0
        while batch_offset < pred.size(0):
            # offset<batch_size

            shape_patches_remaining = shape_patch_count-shape_patch_offset
            batch_patches_remaining = pred.size(0)-batch_offset
            # append estimated patch properties batch to properties for the current shape on the CPU
            num_offset = num
            for i in range(pred.size(0)):
                if not pred[i]:
                    # a = shape_properties[num_offset:num_offset + pred.size(0), :]
                    # b = shape_pol[num_offset:num_offset + pred.size(0)]
                    shape_properties[num_offset+j, :] = originals[batch_offset+i, :]
                    shape_pol[num_offset+j] = pol[batch_offset+i]
                    if opt.label:
                        shape_label[num_offset+j] = labels[batch_offset+i]
                        if labels[i]:
                            noise += 1
                        else:
                            events += 1
                    # c = originals[batch_offset:batch_offset+pred.size(0), :]
                    j += 1
                    num += 1

            if opt.label:
                for i in range(pred.size(0)):
                    o_pred = pred[i]
                    o_labels = labels[i]
                    if o_pred != o_labels:
                        shape_wrong[k,0] = shape_patch_offset+i
                        shape_wrong[k,1:4] = originals[batch_offset+i,:]
                        k += 1
                # record global index and properties of wrong event

                if not noise:
                    SNR = 0
                else:
                    SNR = 10 * math.log(events / noise, 10)
                print('[%s %d/%d] shape %s SNR:%f events:%d noise:%d' % (model_name, batchind, num_batch - 1, dataset.shape_names[shape_ind], SNR, events, noise))
                # SNR score


            batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
            shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)
            if shape_patches_remaining <= batch_patches_remaining:

                if opt.label:
                    normal_prop = shape_properties[0:num, 0:3]
                    labels_prop = shape_label[0:num]
                    labels_prop = labels_prop.numpy()
                    normal_prop = normal_prop.numpy()
                    shape_wrong = shape_wrong[0:k,:]
                    shape_wrong = shape_wrong.numpy()
                    pol_prop = shape_pol[0:num]
                    normal_shape = np.zeros((normal_prop.shape[0], 5))
                    normal_shape[:, 0] = labels_prop
                    normal_shape[:, 1:4] = normal_prop
                    normal_shape[:, 4] = pol_prop
                    np.savetxt(os.path.join(opt.outdir, opt.shapename.rsplit("{", 1)[0] + 'wrong_events.txt'), shape_wrong)
                else:
                    normal_prop = shape_properties[0:num, 0:3]
                    normal_prop = normal_prop.numpy()
                    pol_prop = shape_pol[0:num]
                    normal_shape = np.zeros((normal_prop.shape[0], 4))
                    normal_shape[:, 0:3] = normal_prop
                    normal_shape[:, 3] = pol_prop

                np.save(os.path.join(opt.outdir,opt.shapename.format(i = opt.nrun) + '.npy'), normal_shape)


                # start new shape
                if shape_ind + 1 < len(dataset.shape_names):
                    shape_patch_offset = 0
                    shape_ind = shape_ind + 1
                    if opt.sampling == 'full':
                        shape_patch_count = dataset.shape_patch_count[shape_ind]
                    elif opt.sampling == 'sequential_shapes_random_patches':
                        # shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
                        shape_patch_count = len(datasampler.shape_patch_inds[shape_ind])
                    else:
                        raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
                    shape_properties = torch.FloatTensor(shape_patch_count, 3).zero_()
                    shape_pol = torch.FloatTensor(shape_patch_count).zero_()
                    if opt.label:
                        shape_label = torch.FloatTensor(shape_patch_count).zero_()
                        shape_wrong = torch.FloatTensor(shape_patch_count, 4).zero_()
                        events = 0
                        noise = 0
                        k = 0
                    num = 0


if __name__ == '__main__':
    eval_opt = parse_arguments()
    init_res_directory(eval_opt)
    eval_pcpnet(eval_opt)
