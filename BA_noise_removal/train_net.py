from __future__ import print_function

import argparse
import os
import random
import math
import shutil
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from dataprocess import AEDPatchDataset, RandomAEDPatchSampler, SequentialShapeRandomAEDPatchSampler
from aednet import ResMSAEDNet, ResAEDNet


def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument(
        '--name', type=str, default='AEDNet', help='training run name')
    parser.add_argument(
        '--desc', type=str, default='My training run for AEDNet Background Activity noise removal', help='description')
    parser.add_argument('--indir', type=str, default='../data/AEDNetDataset_BA',
                        help='input folder (events)')
    parser.add_argument('--outdir', type=str, default='../models/BA_noise_removal_model',
                        help='output folder (trained models)')
    parser.add_argument('--logdir', type=str,
                        default='./logs', help='training log folder')
    parser.add_argument('--trainset', type=str,
                        default='training_file_name.txt', help='training set file name')
    parser.add_argument('--testset', type=str,
                        default='test_file_name.txt', help='test set file name')
    parser.add_argument('--saveinterval', type=int,
                        default='1', help='save model each n epochs')
    parser.add_argument('--refine', type=str, default='',
                        help='refine model at this path')

    # training parameters
    parser.add_argument('--nepoch', type=int, default=2000,
                        help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int,
                        default=8, help='input batch size')
    parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n'
                        'point: center point\n'
                        'mean: patch mean')
    parser.add_argument('--patches_per_shape', type=int, default=400,# 800,
                        help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=700,
                        help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int,
                        default=3627473, help='manual seed')
    parser.add_argument('--training_order', type=str, default='random_shape_consecutive', help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=int, default=False,
                        help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='gradient descent momentum')

    # model hyperparameters
    parser.add_argument('--use_point_stn', type=int,
                        default=True, help='use point spatial transformer')
    parser.add_argument('--use_feat_stn', type=int,
                        default=True, help='use feature spatial transformer')
    parser.add_argument('--sym_op', type=str, default='sum',
                        help='symmetry operation')
    parser.add_argument('--points_per_patch', type=int,
                        default=50, help='max. number of points per patch')
    parser.add_argument('--x_frame', type=int,
                        default=1280, help='the width of the frame')
    parser.add_argument('--y_frame', type=int,
                        default=720, help='the height of the frame')
    parser.add_argument('--x_lim', type=float,
                        default=25, help='the limitation of x and y compare to the whole frame')
    parser.add_argument('--y_lim', type=float,
                        default=15, help='the limitation of x and y compare to the whole frame')
    parser.add_argument('--shapenum', type=int,
                        default=0, help='how many points in a shape, if 0 the whole shape was used')


    return parser.parse_args()


def check_path_existance(log_dirname, model_filename, opt):
    if os.path.exists(log_dirname) or os.path.exists(model_filename):
        if os.path.exists(log_dirname):
            shutil.rmtree(os.path.join(opt.logdir, opt.name))


def get_data(opt, train=True, label=True):
    # create train and test dataset loaders
    if train:
        shapes_list_file = opt.trainset
    else:
        shapes_list_file = opt.testset

    dataset = AEDPatchDataset(
        root=opt.indir,
        shapes_list_file=shapes_list_file,
        shape_num=opt.shapenum,
        points_per_patch=opt.points_per_patch,
        x_limitation=opt.x_lim,
        y_limitation=opt.y_lim,
        x_frame=opt.x_frame,
        y_frame=opt.y_frame,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        center=opt.patch_center,
        cache_capacity=opt.cache_capacity,
        label=label)
    if opt.training_order == 'random':
        datasampler = RandomAEDPatchSampler(
            dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        datasampler = SequentialShapeRandomAEDPatchSampler(
            dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    return dataloader, datasampler, dataset


def create_model(n_predicted_features, opt):
    # create model
    aednet = ResAEDNet(
            num_points=opt.points_per_patch,
            output_dim=n_predicted_features,
            use_point_stn=opt.use_point_stn,
            use_feat_stn=opt.use_feat_stn,
            sym_op=opt.sym_op)
    return aednet



def train_pcpnet(opt):
    # colored console output
    def green(x): return '\033[92m' + x + '\033[0m'
    def blue(x): return '\033[94m' + x + '\033[0m'

    log_dirname = os.path.join(opt.logdir, opt.name)
    params_filename = os.path.join(opt.outdir, '%s_params.pth' % (opt.name))
    model_filename = os.path.join(opt.outdir, '%s_model.pth' % (opt.name))
    desc_filename = os.path.join(opt.outdir, '%s_description.txt' % (opt.name))
    checkpoint_filename = os.path.join(opt.outdir, '%s_checkpoint.pth' % (opt.name))

    check_path_existance(log_dirname, model_filename, opt)
    n_predicted_features = 2
    aednet = create_model(n_predicted_features, opt)
    if opt.refine != '':
        aednet.load_state_dict(torch.load(opt.refine))

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)
    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    train_dataloader, train_datasampler, train_dataset = get_data(opt)
    test_dataloader, test_datasampler, test_dataset = get_data(opt)

    opt.train_shapes = train_dataset.shape_names
    opt.test_shapes = test_dataset.shape_names

    print('training set: %d patches (in %d batches) - test set: %d patches (in %d batches)' %
          (len(train_datasampler), len(train_dataloader), len(test_datasampler), len(test_dataloader)))

    try:
        os.makedirs(opt.outdir)
    except OSError:
        pass

    train_writer = SummaryWriter(os.path.join(log_dirname, 'train'))
    test_writer = SummaryWriter(os.path.join(log_dirname, 'test'))

    aednet.cuda()
    optimizer = optim.SGD(aednet.parameters(), lr=opt.lr,
                        momentum=opt.momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50*3200, gamma=0.1)
    start_epoch = -1
    if os.path.exists(model_filename):
        aednet.load_state_dict(torch.load(model_filename))
        checkpoint = torch.load(checkpoint_filename)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    total_train_batches = len(train_dataloader)
    total_test_batches = len(test_dataloader)

    torch.save(opt, params_filename)

    with open(desc_filename, 'w+') as text_file:
        print(opt.desc, file=text_file)

    for epoch in range(start_epoch+1, opt.nepoch):

        current_train_batch_index = -1
        train_completion = 0.0
        train_correct = 0
        train_batches = enumerate(train_dataloader, 0)

        current_test_batch_index = -1
        test_correct = 0
        test_completion = 0.0
        test_batches = enumerate(test_dataloader, 0)
        for current_train_batch_index, data in train_batches:
            scheduler.step(epoch * total_train_batches + current_train_batch_index)
            aednet.train()
            points = data[0]
            points = Variable(points).transpose(2, 1)
            points = points.cuda()

            target = data[1:]
            target = Variable(target[0])
            target = target.cuda()
            optimizer.zero_grad()

            pred, _, _, _ = aednet(points)
            loss, train_correct = compute_loss(pred=pred,
                                target=target,
                                correct=train_correct)

            train_accuracy = train_correct/(opt.batchSize*(current_train_batch_index+1))
            loss.backward()
            optimizer.step()

            train_completion = (current_train_batch_index + 1) / total_train_batches

            # log
            print('[%s %d/%d: %d/%d] %s loss: %f accuracy: %f' % (opt.name, epoch, opt.nepoch, current_train_batch_index,
                                                  total_train_batches - 1, green('train'), loss.item(), train_accuracy))
            # print('min normal len: %f' % (pred.data.norm(2,1).min()))
            train_writer.add_scalar('loss', loss.item(),
                                    (epoch + train_completion) * total_train_batches * opt.batchSize)

            while test_completion <= train_completion and current_test_batch_index + 1 < total_test_batches:

                # 测试模式
                aednet.eval()
                current_test_batch_index, data = next(test_batches)

                points = data[0]
                points = Variable(points, volatile=True)
                points = points.transpose(2, 1)
                points = points.cuda()
                target = data[1:]
                target = Variable(target[0], volatile=True)
                target = target.cuda()

                # 前向传播
                pred, _, _, _ = aednet(points)
                loss, test_correct = compute_loss(
                    pred=pred, target=target,
                    correct=test_correct)

                test_accuracy = test_correct/(opt.batchSize*(current_test_batch_index+1))

                test_completion = (current_test_batch_index + 1) / total_test_batches

                print('[%s %d: %d/%d] %s loss: %f accuracy: %f' % (opt.name, epoch,
                                                      current_train_batch_index, total_train_batches - 1, blue('test'), loss.item(), test_accuracy))
                test_writer.add_scalar(
                    'loss', loss.item(), (epoch + test_completion) *total_train_batches * opt.batchSize)

        # save checkpoint
        checkpoint = {
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        if epoch % opt.saveinterval == 0 or epoch == opt.nepoch - 1:
            torch.save(aednet.state_dict(), model_filename)
            torch.save(checkpoint, checkpoint_filename)

        # save model in a separate file in epochs 0,5,10,50,100,500,1000, ...
        if epoch % (5 * 10**math.floor(math.log10(max(2, epoch - 1)))) == 0 or epoch % 100 == 0 or epoch == opt.nepoch - 1:
            torch.save(aednet.state_dict(), os.path.join(
                opt.outdir, '%s_model_%d.pth' % (opt.name, epoch)))


def compute_labels_loss(pred, output_pred_index, current_output_type_index,  target, output_target_index):
    labels_correct = 0
    o_pred = pred[:, output_pred_index[current_output_type_index]:output_pred_index[current_output_type_index] + 2]  # 8*2网络输出的预测值
    o_target = target[output_target_index[current_output_type_index]].squeeze(1)  # 8*1 labels
    loss_func = torch.nn.CrossEntropyLoss()
    labels_loss = loss_func(o_pred, o_target)
    _, l_pred = torch.max(F.softmax(o_pred), 1)

    for i in range(o_pred.shape[0]):
        if o_target[i] == l_pred[i]:
            labels_correct += 1

    return labels_loss, labels_correct

def compute_loss(pred, target, correct):
    labels_correct = 0
    target = target.squeeze(1)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(pred, target)
    _, l_pred = torch.max(F.softmax(pred), 1)
    for i in range(pred.shape[0]):
        if target[i] == l_pred[i]:
            labels_correct += 1
    correct += labels_correct

    return loss, correct


if __name__ == '__main__':
    train_opt = parse_arguments()
    train_pcpnet(train_opt)
