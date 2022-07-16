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
from dataprocess1 import AERPatchDataset, SequentialAERPatchSampler, SequentialShapeRandomAERPatchSampler
from pcpnet import ResPCPNet
from shutil import copyfile

def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--indir', type=str, default='../data/AERCleanNetTestset_BA', help='input folder (events)')   # 输入路径/data/AERCleanNetTestset
    parser.add_argument('--outdir', type=str, default='./results', help='output folder (estimated events properties)')  # 输出路径results
    parser.add_argument('--dataset', type=str, default='testset.txt', help='shape set file name')  # shape name
    parser.add_argument('--modeldir', type=str, default='../models/BA_noise_removal_model', help='model folder')  # 参数路径
    parser.add_argument('--model', type=str, default='AERCleanNet', help='names of trained models, can evaluate multiple models')  # 模型名称
    parser.add_argument('--modelpostfix', type=str, default='_model.pth', help='model file postfix')  # 模型路径后缀
    parser.add_argument('--parmpostfix', type=str, default='_params.pth', help='parameter file postfix')  # 参数路径后缀
    parser.add_argument('--n_neighbours', type=int, default=200, help='nearest neighbour used for inflation step')  # 临近时间点数
    parser.add_argument('--sampling', type=str, default='full', help='sampling strategy, any of:\n'
                        'full: evaluate all points in the dataset\n'
                        'sequential_shapes_random_patches: pick n random points from each shape as patch centers, shape order is not randomized')  # 采样方式
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)')  # 每个shape多少patch
    parser.add_argument('--seed', type=int, default=40938661, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=0, help='batch size, if 0 the training batch size is used')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--nrun', type=int, default=1, help='nrun')
    parser.add_argument('--shapenum', type=int, default=100000, help='how many points in a shape')
    parser.add_argument('--shapename', type=str, default='dvs71{i}', help='shape to evaluate format : name{i}')
    parser.add_argument('--x_frame', type=int,
                        default=1280, help='the width of the frame')
    parser.add_argument('--y_frame', type=int,
                        default=800, help='the height of the frame')
    parser.add_argument('--x_lim', type=float,
                        default=25, help='the limitation of x and y compare to the whole frame')
    parser.add_argument('--y_lim', type=float,
                        default=15, help='the limitation of x and y compare to the whole frame')



    return parser.parse_args()

def init_res_directory(opt):
    # copy input shape to results directory at first iteration
    if opt.nrun == 1:
        src = os.path.join(opt.indir, opt.shapename.rsplit("{", 1)[0] + ".npy")  # 输入路径中待处理数据的shape名.xyz
        dst = os.path.join(opt.outdir, opt.shapename.format(i = 0) + ".npy" )  # ./result中'name{0}'
        copyfile(src, dst)


def eval_pcpnet(opt):
    # get a list of model names
    model_name = opt.model  # AERCleanNet
    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    model_filename = os.path.join(opt.modeldir, opt.model+opt.modelpostfix)  # 模型路径
    param_filename = os.path.join(opt.modeldir, opt.model+opt.parmpostfix)  # 命令函参数路径

    # load model and training parameters
    trainopt = torch.load(param_filename)  # 加载训练命令行参数
    trainopt.outputs = ['labels']   # 输出为clean_points
    if opt.batchSize == 0:
        model_batchSize = trainopt.batchSize  # batch size与训练相同
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

    pred_dim = 0    # 预测维度
    output_pred_ind = []  # 预测索引值
    for o in trainopt.outputs:
        if o in ['labels']:
            output_pred_ind.append(pred_dim)  # [0]
            pred_dim += 2  # 3
        else:
            raise ValueError('Unknown output: %s' % (o))
    dataset = AERPatchDataset(
        root=opt.outdir, shapes_list_file=opt.dataset,
        shape_num=opt.shapenum,
        points_per_patch=trainopt.points_per_patch,
        patch_features=['original'],
        x_limitation=x_lim,
        y_limitation=y_lim,
        x_frame=x_frame,
        y_frame=y_frame,
        seed=opt.seed,
        use_pca=trainopt.use_pca,
        center=trainopt.patch_center,
        point_tuple=trainopt.point_tuple,
        cache_capacity=opt.cache_capacity,
        shape_names = [opt.shapename.format(i = opt.nrun-1)],
        train=False)
    # 根目录/result 文件名 testset.txt
    if opt.sampling == 'full':
        datasampler = SequentialAERPatchSampler(dataset)
    elif opt.sampling == 'sequential_shapes_random_patches':
        datasampler = SequentialShapeRandomAERPatchSampler(
            dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            sequential_shapes=True,
            identical_epochs=False)
    else:
        raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
    # 'full'按顺序提取 'sequential_shapes_random_patches'按顺序每个shape提取patches_per_shape
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=datasampler,
        batch_size=model_batchSize,
        num_workers=int(opt.workers))
    # 数据提取

    regressor = ResPCPNet(
        num_points=trainopt.points_per_patch,
        output_dim=pred_dim,
        use_point_stn=trainopt.use_point_stn,
        use_feat_stn=trainopt.use_feat_stn,
        sym_op=trainopt.sym_op,
        point_tuple=trainopt.point_tuple)
    # 网络：500个点 输出3维 使用刚性归一，特征归一 最大池化层 tuple=1
    regressor.load_state_dict(torch.load(model_filename))  # 加载网络参数
    regressor.cuda()

    shape_ind = 0  # shape索引
    shape_patch_offset = 0
    if opt.sampling == 'full':
        n = len(dataset.shape_patch_count)
        shape_patch_count = (n-1)*opt.shapenum+dataset.shape_patch_count[n-1]  # patch数=包含点数
    elif opt.sampling == 'sequential_shapes_random_patches':
        shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
    else:
        raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
    shape_properties = torch.FloatTensor(shape_patch_count, 3).zero_()  # patch数*3
    shape_pol = torch.FloatTensor(shape_patch_count).zero_()

    # append model name to output directory and create directory if necessary
    model_outdir = os.path.join(opt.outdir, model_name)  # 模型参数路径在./result/AERcleanNet
    if not os.path.exists(model_outdir):
        os.makedirs(model_outdir)  # 创建模型路径

    num_batch = len(dataloader)  # batch数目

    batch_enum = enumerate(dataloader, 0)

    num = 0
    k = 0

    regressor.eval()
    for batchind, data in batch_enum:

        # get batch, convert to variables and upload to GPU
        points, originals, pol, data_trans = data  # 取出临近点，中心点的值，灰度，旋转矩阵

        points = Variable(points, volatile=True)  # 设置点为变量 且不会进行反向传播 8*500*3
        points = points.transpose(2, 1)  # 8*3*500
        points = points.cuda()
        originals = originals.cuda()
        pol = pol.cuda()

        pred, _, _,_ = regressor(points)  # 经过网络输出得到预测值和旋转矩阵

        pred = pred.data
        _, pred = torch.max(F.softmax(pred, dim=1), 1)

        print('[%s %d/%d] shape %s' % (model_name, batchind, num_batch-1, dataset.shape_names[shape_ind]))

        batch_offset = 0
        j = 0
        while batch_offset < pred.size(0):
            # offset<batch_size

            shape_patches_remaining = shape_patch_count-shape_patch_offset  # 当前所剩patch
            batch_patches_remaining = pred.size(0)-batch_offset  # 当前batch所剩
            # append estimated patch properties batch to properties for the current shape on the CPU
            num_offset = num
            for i in range(pred.size(0)):
                if not pred[i]:
                    a = shape_properties[num_offset:num_offset + pred.size(0), :]
                    shape_properties[num_offset+j, :] = originals[batch_offset+i, :]
                    shape_pol[num_offset+j] = pol[batch_offset+i]
                    c = originals[batch_offset:batch_offset+pred.size(0), :]
                    j += 1
                    num += 1
            # 将batch里的数据加入到整个shape数据中


            batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)  # batchsize/最后一次所剩batch
            shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)  # 处理的patch计数

            if shape_patches_remaining <= batch_patches_remaining:
                # 最后一个batch

                # save shape properties to disk
                prop_saved = [False]*len(trainopt.outputs)  # [False]

                # save clean points
                oi = [k for k, o in enumerate(trainopt.outputs) if o in ['labels']]  # 训练output中'clean_points'的索引
                if len(oi) > 1:
                    raise ValueError('Duplicate point output.')
                elif len(oi) == 1:
                    oi = oi[0]
                    normal_prop = shape_properties[0:num, output_pred_ind[oi]:output_pred_ind[oi]+3]  # 取出修正后的shape
                    # Compute mean displacements, inspired from Taubin smoothing
                    normal_prop = normal_prop.numpy()
                    pol_prop = shape_pol[0:num]
                    normal_shape = np.zeros((normal_prop.shape[0], 4))
                    normal_shape[:, 0:3] = normal_prop
                    normal_shape[:, 3] = pol_prop
                    np.save(os.path.join(opt.outdir,opt.shapename.format(i = opt.nrun) + '.npy'), normal_shape) # 存成./results/name[1].xyz（txt形式）
                    prop_saved[oi] = True

                if not all(prop_saved):
                    raise ValueError('Not all shape properties were saved, some of them seem to be unsupported.')
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
                    num = 0

if __name__ == '__main__':
    eval_opt = parse_arguments()
    init_res_directory(eval_opt)
    eval_pcpnet(eval_opt)
