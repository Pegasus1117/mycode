from importlib import import_module

from easydict import EasyDict as edict
import torch.nn as nn
import torch

from lib.core import init_torch, load_weights
from lib.util import pickle_read
from models import resnet_dilate
import os

from models.deform_conv_v2 import DeformConv2d

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



class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()

        path = os.path.dirname(os.getcwd())  # 加入到mytrain中,os.getcwd()会发生变化,到时候得修改;
        conf_path = os.path.join(path, 'pretrain/conf.pkl')  # 得修改;
        weights_path = os.path.join(path, 'pretrain/model_40000_pkl')  # 得修改;

        conf = edict(pickle_read(conf_path))
        # print("conf: \n", conf)

        init_torch(conf.rng_seed, conf.cuda_seed)

        srcnet = import_module('models.' + conf.model).build(conf)
        load_weights(srcnet, weights_path, remove_module=True)
        # print("basenet: \n", basenet)

        self.base = srcnet.base
        self.depthnet = srcnet.depthnet

        self.adaptive_diated = conf.adaptive_diated
        self.dropout_position = conf.dropout_position  # 'early'   # 'early'  'late' 'adaptive'
        self.use_dropout = conf.use_dropout  # True
        self.drop_channel = conf.drop_channel  # True
        self.use_corner = conf.use_corner  # False
        self.corner_in_3d = conf.corner_in_3d  # False
        self.deformable = conf.deformable

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

        self.prop_feats = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )             # [N,2018,H,W]->[N,512,H,W]
        if self.use_dropout:    # True
            self.dropout = nn.Dropout(p=conf.dropout_rate)

        if self.drop_channel:   # True
            self.dropout_channel = nn.Dropout2d(p=0.3)  # 通常输入为 nn.Conv2d modules.

        self.occlusion = nn.Conv2d(self.prop_feats[0].out_channels, 7, 1)



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




if __name__ == "__main__":
    net = Mynet()
    print("net:\n", net)
