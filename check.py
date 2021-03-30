import numpy as np
# from lib.util import *
import os
# import torch.nn as nn
# from torchvision import models
import cv2

def iou():
    box_a = np.array([[1.5, 1.5, 2.5, 2.5], [2, 2.5, 3, 3.5], [2.5, 2.5, 3.5, 3.5]])
    box_b = np.array([[2.0, 2, 3, 3], [3.0, 3, 4, 4]])
    print(box_a)
    print(box_b)
    max_xy = np.minimum(box_a[:, 2:4], np.expand_dims(box_b[:, 2:4], axis=1))
    min_xy = np.maximum(box_a[:, 0:2], np.expand_dims(box_b[:, 0:2], axis=1))
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

    result = inter[:, :, 0] * inter[:, :, 1]
    print("result: \n", result)

    return None

def size():
    scale = 1
    test_scale = 512
    imH = 370
    imW = 1224
    feat_stride = 16

    scale_factor = scale * test_scale / imH
    res = np.array([imH, imW]) * scale_factor
    feat_size = np.ceil(np.array(res) / feat_stride).astype(int)
    print(feat_size)

def anchor_center(w, h, stride):
    """
    Centers an anchor based on a stride and the anchor shape (w, h).

    center ground truths with steps of half stride
    hence box 0 is centered at (7.5, 7.5) rather than (0, 0)
    for a feature stride of 16 px.
    """

    anchor = np.zeros([4], dtype=np.float32)

    anchor[0] = -w / 2 + (stride - 1) / 2
    anchor[1] = -h / 2 + (stride - 1) / 2
    anchor[2] = w / 2 + (stride - 1) / 2
    anchor[3] = h / 2 + (stride - 1) / 2

    return anchor

def anchors():
    min_gt_h = 32.0
    max_gt_h = 384.0
    base = (max_gt_h / min_gt_h) ** (1 / (12 - 1))
    anchor_scales = np.array([min_gt_h * (base ** i) for i in range(0, 12)])
    anchor_ratios = np.array([0.5, 1.0, 1.5])
    anchors = np.zeros([len(anchor_scales) * len(anchor_ratios), 4], dtype=np.float32)  # (36,4)

    aind = 0

    # compute simple anchors based on scale/ratios
    # 生成不同大小和比例的simple anchors；(36,4): 共36种（12类大小＊3类长宽比例）,四个值存放的为anchor框的对角坐标
    for scale in anchor_scales:

        for ratio in anchor_ratios:
            h = scale
            w = scale * ratio

            anchors[aind, 0:4] = anchor_center(w, h, 16)  # anchors的前四个值存放的为anchor框的对角坐标
            aind += 1

    print(anchors)

def pkl_read():
    basepath = os.getcwd()  # /home/lab316/Documents/Code/D4LCN
    # print(basepath)
    filename = 'bbox_means.pkl'    # 'bbox_means.pkl' , 'bbox_stds.pkl' , 'anchors.pkl' , 'imdb.pkl'
    filepath = os.path.join(basepath, 'output', 'depth_guided_config', filename)

    file = pickle_read(filepath)
    print(file)

def nptest():
    a = np.arange(12).reshape(3, 4)
    print(a)
    b = np.array([[1, 2, 3, 4, 5]])
    print(b)
    print(b[:, 1])
    print(b[:, 1][0])
    a_1 = a * b[:, 1]
    print(a_1)
    a_1 = a * b[:, 1][0]
    print(a_1)

def image_read():
    image = cv2.imread("/media/hyj/ElementsSE/kitti/training/image_2/000010.png")
    cv2.imshow("image", image)
    # cv2.waitKey(0)

    for i in range(int(image.shape[2] / 3)):
        # convert to RGB then permute to be [B C H W]
        image[:, :, (i * 3):(i * 3) + 3] = image[:, :, (i * 3 + 2, i * 3 + 1, i * 3)]
    cv2.imshow("image_converted", image)
    cv2.waitKey(0)

'''
class ResNetDilate(nn.Module):  # 没有最后全连接层和池化层的预训练好参数的ResNet-50
    def __init__(self, num_layer=50):
        super(ResNetDilate, self).__init__()
        # ==========1.先把整个预先训练好的网络提取出来==========
        if num_layer == 50:
            model_resnet = models.resnet50(pretrained=True)
        if num_layer == 101:
            model_resnet = models.resnet101(pretrained=True)
        # ==========2.再把从整个网络截取用到的网络层==========
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4

        for n, m in self.layer4.named_modules():  # TOdo : 还需要观察变化
            if 'conv2' in n:  # conv1 for resnet34
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)  # 与原来相比：1 -> (2,2);1 -> (2,2);1 -> (1,1);
            elif 'downsample.0' in n:
                m.stride = (1, 1)  # 与原来相比： 1 -> (1,1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def ModulePrint():
    print("resnet原模型：")
    model = models.resnet50(pretrained=True)
    print(model)
    print("resnet现模型：")
    net = ResNetDilate()
    print(net)
'''

if __name__ == "__main__" :
    # iou()
    # size()
    # anchors()
    # pkl_read()
    # ModulePrint()
    # nptest()
    image_read()