import torch.nn as nn
from lib.rpn_util import *
from models import resnet
import torch
import numpy as np
from models.deform_conv_v2 import *


def dynamic_local_filtering(x, depth, dilated=1):
    padding = nn.ReflectionPad2d(dilated)  # ConstantPad2d(1, 0)    # 以输入边界为法平面的镜像数据填充；
    pad_depth = padding(depth)
    n, c, h, w = x.size()
    # y = torch.cat((x[:, int(c/2):, :, :], x[:, :int(c/2), :, :]), dim=1)
    # x = x + y
    y = torch.cat((x[:, -1:, :, :], x[:, :-1, :, :]), dim=1)    # 对应论文中shift-pooling operator中shift步长为1；
    z = torch.cat((x[:, -2:, :, :], x[:, :-2, :, :]), dim=1)    # 对应论文中shift-pooling operator中shift步长为2；
    x = (x + y + z) / 3
    pad_x = padding(x)
    filter = (pad_depth[:, :, dilated: dilated + h, dilated: dilated + w] * pad_x[:, :, dilated: dilated + h, dilated: dilated + w]).clone()    # 9次变换最中心的那次变换；
    for i in [-dilated, 0, dilated]:
        for j in [-dilated, 0, dilated]:
            if i != 0 or j != 0:
                filter += (pad_depth[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w] * pad_x[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w]).clone()
    return filter / 9

class RPN(nn.Module):


    def __init__(self, phase, conf):
        super(RPN, self).__init__()

        self.base = resnet.ResNetDilate(conf.base_model)    # 去掉最后全连接层和池化层并修改layer4中部分网络的预训练好参数的ResNet-50,作为特征提取网络的主干
        self.adaptive_diated = conf.adaptive_diated         # True
        self.dropout_position = conf.dropout_position       # 'early'   # 'early'  'late' 'adaptive'
        self.use_dropout = conf.use_dropout                 # True
        self.drop_channel = conf.drop_channel               # True
        self.use_corner = conf.use_corner                   # False
        self.corner_in_3d = conf.corner_in_3d               # False
        self.deformable = conf.deformable                   # False

        if conf.use_rcnn_pretrain:  # False
            # print(self.base.state_dict().keys())
            if conf.base_model == 101:
                pretrained_model = torch.load('faster_rcnn_1_10_14657.pth')['model']
                rename_dict = {'RCNN_top.0': 'layer4', 'RCNN_base.0': 'conv1', 'RCNN_base.1': 'bn1', 'RCNN_base.2': 'relu',
                               'RCNN_base.3': 'maxpool', 'RCNN_base.4': 'layer1',
                               'RCNN_base.5': 'layer2', 'RCNN_base.6': 'layer3'}
                change_dict = {}
                for item in pretrained_model.keys():
                    for rcnn_name in rename_dict.keys():
                        if rcnn_name in item:
                            change_dict[item] = item.replace(rcnn_name, rename_dict[rcnn_name])
                            break
                pretrained_model = {change_dict[k]: v for k, v in pretrained_model.items() if k in change_dict}
                self.base.load_state_dict(pretrained_model)

            elif conf.base_model == 50:
                pretrained_model = torch.load('res50_faster_rcnn_iter_1190000.pth',
                                              map_location=lambda storage, loc: storage)
                pretrained_model = {k.replace('resnet.', ''): v for k, v in pretrained_model.items() if 'resnet' in k}
                # print(pretrained_model.keys())
                self.base.load_state_dict(pretrained_model)


        self.depthnet = resnet.ResNetDilate(50)

        if self.adaptive_diated:    # True
            self.adaptive_softmax = nn.Softmax(dim=3)

            self.adaptive_layers = nn.Sequential(
                nn.AdaptiveMaxPool2d(3),    # 常见的池化参数为kernel_size,‘’图像‘’的输出尺寸另算（可变）；AdaptiveMaxPool2d()参数为output_size，对于任何输入大小，其‘’图像‘’输出尺寸就是output_siz；
                nn.Conv2d(512, 512 * 3, 3, padding=0),
            )                               # 用在layer2与layer3之间;    # size: ->(512,512*3,1,1)
            self.adaptive_bn = nn.BatchNorm2d(512)
            self.adaptive_relu = nn.ReLU(inplace=True)

            self.adaptive_layers1 = nn.Sequential(
                nn.AdaptiveMaxPool2d(3),
                nn.Conv2d(1024, 1024 * 3, 3, padding=0),
            )                               # # 用在layer3与layer4之间;  # size: ->(1024,1024*3,1,1)
            self.adaptive_bn1 = nn.BatchNorm2d(1024)
            self.adaptive_relu1 = nn.ReLU(inplace=True)

        if self.deformable:     # False
            self.deform_layer = DeformConv2d(512, 512, 3, padding=1, bias=False, modulation=True)

        # settings
        self.phase = phase
        self.num_classes = len(conf['lbls']) + 1    # +1的理解：不属于3个中的任何一类？
        self.num_anchors = conf['anchors'].shape[0]

        self.prop_feats = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )             # [N,2018,H,W]->[N,512,H,W]
        if self.use_dropout:    # True
            self.dropout = nn.Dropout(p=conf.dropout_rate)

        if self.drop_channel:   # True
            self.dropout_channel = nn.Dropout2d(p=0.3)  # 通常输入为 nn.Conv2d modules.

        # outputs
        self.cls = nn.Conv2d(self.prop_feats[0].out_channels, self.num_classes * self.num_anchors, 1)   # [N,512,H,W]->[N,36*4,H,W]

        # bbox 2d
        self.bbox_x = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)   # # [N,512,H,W]->[N,36,H,W]
        self.bbox_y = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        # bbox 3d
        self.bbox_x3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_z3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_l3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_rY3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        if self.corner_in_3d:   # False
            self.bbox_3d_corners = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors * 18, 1)  # 2 * 8 + 2
            self.bbox_vertices = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors * 24, 1)  # 3 * 8
        elif self.use_corner:   # False
            self.bbox_vertices = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors * 24, 1)

        self.softmax = nn.Softmax(dim=1)

        self.feat_stride = conf.feat_stride     # 16
        self.feat_size = calc_output_size(np.array(conf.crop_size), self.feat_stride)   # feat_size=[32,106]；
        self.rois = locate_anchors(conf.anchors, self.feat_size, conf.feat_stride, convert_tensor=True)
        self.rois = self.rois.type(torch.cuda.FloatTensor)
        self.anchors = conf.anchors

    def forward(self, x, depth):

        batch_size = x.size(0)

        x = self.base.conv1(x)
        depth = self.depthnet.conv1(depth)
        x = self.base.bn1(x)
        depth = self.depthnet.bn1(depth)
        x = self.base.relu(x)
        depth = self.depthnet.relu(depth)
        x = self.base.maxpool(x)
        depth = self.depthnet.maxpool(depth)

        x = self.base.layer1(x)
        depth = self.depthnet.layer1(depth)
        # x = dynamic_local_filtering(x, depth, dilated=1) + dynamic_local_filtering(x, depth, dilated=2) + dynamic_local_filtering(x, depth, dilated=3)

        x = self.base.layer2(x)
        depth = self.depthnet.layer2(depth)

        if self.deformable:     # False
            depth = self.deform_layer(depth)
            x = x * depth

        if self.adaptive_diated:    # True      # 加了一层adaptive_layers()，作用是产生权重，作为3种dilated的权重；
            weight = self.adaptive_layers(x).reshape(-1, 512, 1, 3)     # size: (512,512*3,1,1)->(512,512,1,3)
            weight = self.adaptive_softmax(weight)
            x = dynamic_local_filtering(x, depth, dilated=1) * weight[:, :, :, 0:1] \
                + dynamic_local_filtering(x, depth, dilated=2) * weight[:, :, :, 1:2] \
                + dynamic_local_filtering(x, depth, dilated=3) * weight[:, :, :, 2:3]
            x = self.adaptive_bn(x)
            x = self.adaptive_relu(x)
        else:
            x = dynamic_local_filtering(x, depth, dilated=1) + dynamic_local_filtering(x, depth, dilated=2) + dynamic_local_filtering(x, depth, dilated=3)

        if self.use_dropout and self.dropout_position == 'adaptive':
            x = self.dropout(x)

        if self.drop_channel:   # True
            x = self.dropout_channel(x)

        x = self.base.layer3(x)
        depth = self.depthnet.layer3(depth)

        if self.adaptive_diated:
            weight = self.adaptive_layers1(x).reshape(-1, 1024, 1, 3)
            weight = self.adaptive_softmax(weight)
            x = dynamic_local_filtering(x, depth, dilated=1) * weight[:, :, :, 0:1] \
                + dynamic_local_filtering(x, depth, dilated=2) * weight[:, :, :, 1:2] \
                + dynamic_local_filtering(x, depth, dilated=3) * weight[:, :, :, 2:3]
            x = self.adaptive_bn1(x)
            x = self.adaptive_relu1(x)
        else:
            x = x * depth

        x = self.base.layer4(x)
        depth = self.depthnet.layer4(depth)
        x = x * depth

        if self.use_dropout and self.dropout_position == 'early':
            x = self.dropout(x)

        prop_feats = self.prop_feats(x)  # 加了一层卷积的理解：它连接了主干网络特征提取与头部预测网络，可能是为了调整特征图的尺寸，以便接入后续的输入统一的头部预测网络

        if self.use_dropout and self.dropout_position == 'late':
            prop_feats = self.dropout(prop_feats)

        cls = self.cls(prop_feats)

        # bbox 2d
        bbox_x = self.bbox_x(prop_feats)
        bbox_y = self.bbox_y(prop_feats)
        bbox_w = self.bbox_w(prop_feats)
        bbox_h = self.bbox_h(prop_feats)

        # bbox 3d
        bbox_x3d = self.bbox_x3d(prop_feats)
        bbox_y3d = self.bbox_y3d(prop_feats)
        bbox_z3d = self.bbox_z3d(prop_feats)
        bbox_w3d = self.bbox_w3d(prop_feats)
        bbox_h3d = self.bbox_h3d(prop_feats)
        bbox_l3d = self.bbox_l3d(prop_feats)
        bbox_rY3d = self.bbox_rY3d(prop_feats)
        # targets_dx, targets_dy, delta_z, scale_w, scale_h, scale_l, deltaRotY

        feat_h = cls.size(2)    # 网络输出的特征图的高H
        feat_w = cls.size(3)    # 网络输出的特征图的宽W

        # reshape for cross entropy
        cls = cls.view(batch_size, self.num_classes, feat_h * self.num_anchors, feat_w)  # [N,36*4,H,W]->[N,4,36*H,W]

        # score probabilities
        prob = self.softmax(cls)    # [N,4,36*H,W]

        # reshape for consistency
        # although it's the same with x.view(batch_size, -1, 1) when c == 1, useful when c > 1
        bbox_x = flatten_tensor(bbox_x.view(batch_size, 1, feat_h * self.num_anchors, feat_w))  # [N,36,H,W]->[N,1,36*H,W]->[N,36*H*W,1];
        bbox_y = flatten_tensor(bbox_y.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w = flatten_tensor(bbox_w.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h = flatten_tensor(bbox_h.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        bbox_x3d = flatten_tensor(bbox_x3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w)) # [N,36,H,W]->[N,1,36*H,W]->[N,36*H*W,1];
        bbox_y3d = flatten_tensor(bbox_y3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_z3d = flatten_tensor(bbox_z3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w3d = flatten_tensor(bbox_w3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h3d = flatten_tensor(bbox_h3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_l3d = flatten_tensor(bbox_l3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_rY3d = flatten_tensor(bbox_rY3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        # bundle    打包
        bbox_2d = torch.cat((bbox_x, bbox_y, bbox_w, bbox_h), dim=2)    # [N,36*H*W,4]
        bbox_3d = torch.cat((bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_rY3d), dim=2) # [N,36*H*W,7]

        if self.corner_in_3d:
            corners_3d = self.bbox_3d_corners(prop_feats)
            corners_3d = flatten_tensor(corners_3d.view(batch_size, 18, feat_h * self.num_anchors, feat_w))
            bbox_vertices = self.bbox_vertices(prop_feats)
            bbox_vertices = flatten_tensor(bbox_vertices.view(batch_size, 24, feat_h * self.num_anchors, feat_w))
        elif self.use_corner:
            bbox_vertices = self.bbox_vertices(prop_feats)
            bbox_vertices = flatten_tensor(bbox_vertices.view(batch_size, 24, feat_h * self.num_anchors, feat_w))

        feat_size = [feat_h, feat_w]

        cls = flatten_tensor(cls)    # [N,4,36*H,W]->[N,36*H*W,4]   #TOdo: 尺寸问题？？
        prob = flatten_tensor(prob)  # [N,4,36*H,W]->[N,36*H*W,4]

        if self.training:
            #print(cls.size(), prob.size(), bbox_2d.size(), bbox_3d.size(), feat_size)
            if self.corner_in_3d:
                return cls, prob, bbox_2d, bbox_3d, torch.from_numpy(np.array(feat_size)).cuda(), bbox_vertices, corners_3d
            elif self.use_corner:
                return cls, prob, bbox_2d, bbox_3d, torch.from_numpy(np.array(feat_size)).cuda(), bbox_vertices
            else:
                return cls, prob, bbox_2d, bbox_3d, torch.from_numpy(np.array(feat_size)).cuda()

        else:

            if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
                self.feat_size = [feat_h, feat_w]
                self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride, convert_tensor=True)
                self.rois = self.rois.type(torch.cuda.FloatTensor)

            return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois


def build(conf, phase='train'):

    train = phase.lower() == 'train'

    rpn_net = RPN(phase, conf)
    print(rpn_net)
    if train: rpn_net.train()   #
    else: rpn_net.eval()

    return rpn_net
