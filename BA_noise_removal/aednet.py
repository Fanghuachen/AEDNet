from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

# basic module for residual block
class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, conv = False):
        super(BasicBlock, self).__init__()
        if conv:
            self.l1 = torch.nn.Conv1d(in_planes, planes, 5, padding=2)
            self.l2 = torch.nn.Conv1d(planes, planes, 5, padding=2)
        else:
            self.l1 = nn.Linear(in_planes, planes)
            self.l2 = nn.Linear(planes, planes)

        stdv = 0.1 # for initialisation

        self.l1.weight.data.uniform_(-stdv, stdv)
        self.l2.weight.data.uniform_(-stdv, stdv)
        self.l1.bias.data.uniform_(-stdv, stdv)
        self.l2.bias.data.uniform_(-stdv, stdv)

        self.bn1 = nn.BatchNorm1d(planes, momentum = 0.1)
        self.shortcut = nn.Sequential()
        if in_planes != planes:
            if conv:
                self.l0 = nn.Conv1d(in_planes, planes, 9, padding=4)
            else:
                self.l0 = nn.Linear(in_planes, planes)

            self.l0.weight.data.uniform_(-stdv, stdv)
            self.l0.bias.data.uniform_(-stdv, stdv)

            self.shortcut = nn.Sequential(self.l0, nn.BatchNorm1d(planes))
        self.bn2 = nn.BatchNorm1d(planes, momentum = 0.1)

    def forward(self, x):
            out = F.relu(self.bn1(self.l1(x)))
            out = self.bn2(self.l2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


class ResSTN(nn.Module):
    def __init__(self, num_scales=1, num_points=50, dim=2, sym_op='sum'):
        super(ResSTN, self).__init__()
        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.b1 = BasicBlock(self.dim, 64, conv = True)
        self.b2 = BasicBlock(64, 128, conv = True)
        self.b3 = BasicBlock(128, 1024, conv = True)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.bfc1 = BasicBlock(1024, 512)
        self.bfc2 = BasicBlock(512, 256)
        self.bfc3 = BasicBlock(256, self.dim*self.dim)

        if self.num_scales > 1:
            self.bfc0 = BasicBlock(1024*self.num_scales, 1024)


    def forward(self, x):
        batchsize = x.size()[0]
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        # symmetric operation over all points
        if self.num_scales == 1:
            if self.sym_op == 'max':
                x = self.mp1(x)
            elif self.sym_op == 'sum':
                x = torch.sum(x, 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
        else:
            if x.is_cuda:
                x_scales = Variable(torch.cuda.FloatTensor(x.size(0), 1024*self.num_scales, 1))
            else:
                x_scales = Variable(torch.FloatTensor(x.size(0), 1024*self.num_scales, 1))
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = self.bfc0(x)

        x =self.bfc1(x)
        x = self.bfc2(x)
        x = self.bfc3(x)


        iden = Variable(torch.from_numpy(np.identity(self.dim, 'float32')).clone()).view(1, self.dim*self.dim).repeat(batchsize, 1)

        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.dim, self.dim)

        return x


class ResEventNetfeat(nn.Module):
    def __init__(self, num_scales=1, num_points=50, use_point_stn=True, use_feat_stn=True, sym_op='sum', get_pointfvals=False):
        super(ResEventNetfeat, self).__init__()
        self.num_points = num_points
        self.num_scales = num_scales
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.get_pointfvals = get_pointfvals

        if self.use_point_stn:
            self.stn1 = ResSTN(num_scales=self.num_scales, num_points=num_points, dim=2, sym_op=self.sym_op)

        if self.use_feat_stn:
            self.stn2 = ResSTN(num_scales=self.num_scales, num_points=num_points, dim=64, sym_op=self.sym_op)

        self.b0a = BasicBlock(2, 64, conv = True)
        self.b0b = BasicBlock(64, 64, conv=True)

        self.b1 = BasicBlock(64, 64, conv = True)
        self.b2 = BasicBlock(64, 128, conv = True)
        self.b3 = BasicBlock(128, 1024, conv = True)

        if self.num_scales > 1:
            self.b4 = BasicBlock(1024, 1024*self.num_scales, conv = True)

        if self.sym_op == 'max':
            self.mp1 = torch.nn.MaxPool1d(num_points)
        elif self.sym_op == 'sum':
            self.mp1 = None
        else:
            raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

    def forward(self, x):

        # input transform
        if self.use_point_stn:
            trans = self.stn1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None

        # feature extraction (3,64)
        x = self.b0a(x)
        x = self.b0b(x)

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None

        # feature extraction (64,128,1024)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)

        # feature extraction (1024,1024*num_scales)
        if self.num_scales > 1:
            x = self.b4(x)

        if self.get_pointfvals:
            pointfvals = x
        else:
            pointfvals = None
            # so the intermediate result can be forgotten if it is not needed

        # symmetric sum operation over all points
        if self.num_scales == 1:
            if self.sym_op == 'max':
                x = self.mp1(x)
            elif self.sym_op == 'sum':
                x = torch.sum(x, 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

        else:
            if x.is_cuda:
                x_scales = Variable(torch.cuda.FloatTensor(x.size(0), 1024*self.num_scales**2, 1))
            else:
                x_scales = Variable(torch.FloatTensor(x.size(0), 1024*self.num_scales**2, 1))
            if self.sym_op == 'max':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*1024:(s+1)*self.num_scales*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            elif self.sym_op == 'sum':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*1024:(s+1)*self.num_scales*1024, :] = torch.sum(x[:, :, s*self.num_points:(s+1)*self.num_points], 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
            x = x_scales

        x = x.view(-1, 1024*self.num_scales**2)

        return x, trans, trans2, pointfvals


class ResAEDNet(nn.Module):
    def __init__(self, num_points=50, output_dim=2, use_point_stn=True, use_feat_stn=True, sym_op='sum', get_pointfvals=False):
        super(ResAEDNet, self).__init__()
        self.num_points = num_points

        self.feat = ResEventNetfeat(
            num_points=num_points,
            num_scales=1,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals)

        self.b1 = BasicBlock(1024, 512)

        self.b2 = BasicBlock(512, 256)
        self.b3 = BasicBlock(256, output_dim)


    def forward(self, x):
        x, trans, trans2, pointfvals = self.feat(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return x, trans, trans2, pointfvals

class ResMSAEDNet(nn.Module):
    def __init__(self, num_scales=2, num_points=500, output_dim=3, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=False):
        super(ResMSAEDNet, self).__init__()
        self.num_points = num_points

        self.feat = ResEventNetfeat(
            num_points=num_points,
            num_scales=num_scales,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals)
        self.b0 = BasicBlock(1024*num_scales**2, 1024)
        self.b1 = BasicBlock(1024, 512)
        self.b2 = BasicBlock(512, 256)
        self.b3 = BasicBlock(256, output_dim)

    def forward(self, x):
        x, trans, trans2, pointfvals = self.feat(x)
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return x, trans, trans2, pointfvals

