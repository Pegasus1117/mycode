from time import time

import numpy as np
# from lib.util import *
import os
# import torch.nn as nn
# from torchvision import models
import cv2
import torch

from lib.core import iou
from lib.rpn_util import bbXYWH2Coords, bbCoords2XYWH


def iou_test():
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
    filename = 'bbox_means.pkl'  # 'bbox_means.pkl' , 'bbox_stds.pkl' , 'anchors.pkl' , 'imdb.pkl'
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


class Filtrate():
    # 测试数据
    testset = np.array([[0, 0, 0, 0, 0, 'a', True],
                        [1, 1, 1, 1, 1, 'b', False],
                        [6, 6, 6, 6, 6, 'c', True],
                        [3, 3, 3, 3, 3, 'd', False],
                        [4, 4, 4, 4, 4, 'e', True],
                        [2, 2, 2, 2, 2, 'c', True]
                        ])
    indset = np.array([[True],
                       [False],
                       [True],
                       [False],
                       [True],
                       [True]
                       ])

    # print(testset)
    # print(testset.shape)
    # print(indset.shape)

    def filter(testset=None):
        return testset

    def sort(testset=None):
        return testset

    def modify_test(testset=None):
        # 一维数组
        print("对一维数组使用argsort()进行排序后修改:")
        ns = np.array([1, 5, 5, 2, 6, 8, 8, 6, 9, 0])
        print("数组排序前:", ns)
        print("数组排序后:", np.sort(ns))

        id = ns.argsort()
        print("排序后索引:", id)
        print("排序后原数组的最小值:", ns[id[0]])

        ns[id[0]] += 1
        print("修改最小值:", ns)

        print("按顺序提取打印前两个值:")
        for i in range(2):
            print(ns[id[i]])
        print("操作后数组:", ns)

        # 二维数组
        print("\n=================================\n")
        print("对二维数组使用argsort()进行排序后修改:")
        print("方法一:提取出作为判断的某列后用argsort:")
        print("输入数据:\n", testset)

        judg = testset[:, 0]
        print("排序依据:", judg)
        id = judg.argsort()
        print("排序后索引:", id)

        print("直接提取出第4小的数据:", testset[id[np.sort(id)[3]]])
        # for i in id :
        #     print(testset[i,:])

        print("---------------------------------")
        print("方法二:直接用argsort:")
        ns = np.array([[2, 4, 6], [5, 3, 1], [0, 7, -1]])
        print("ns:\n", ns)

        id = ns.argsort(axis=1)
        print("axis= 0:\n", id)
        print("按第二列的数据大小进行排序:\n", ns[id[:, 1]])

        ns[id[:, 1][0]] = 0
        print("按第二列的数据大小进行排序,然后修改最小的一行:\n", ns[id[:, 1]])

        ns[id[:, 1][0]][0] = -1
        print("按第二列的数据大小进行排序,然后修改最小的一行的第0个值:\n", ns[id[:, 1]])

        ns[id[:, 1][0]][0:2] = -2
        print("按第二列的数据大小进行排序,然后修改最小的一行的第0-1个值:\n", ns[id[:, 1]])

        # 三维数组
        print("\n=================================\n")
        print("对三维数组使用argsort()进行排序后修改:")
        ns = np.array([[[2, 4, 6], [5, 3, 1], [0, 7, -1]],
                       [[0 * 3, 7 * 3, -1 * 3], [2 * 3, 4 * 3, 6 * 3], [5 * 3, 3 * 3, 1 * 3]]])
        print("数据:\n", ns)
        print("方法一:提取出作为判断的某列后用argsort:")
        judge = ns[:, :, 1]
        print("排序依据数据:\n", judge)
        print("显然这种方法不太合适")
        print("---------------------------------")
        print("方法二:直接用argsort:")
        print("数据:\n", ns)

        id = ns.argsort(axis=1)
        print("id:\n", id)
        print("id[:, :, 1]:\n", id[:, :, 1])
        print("按第二列的数据大小进行排序:\n", ns[0, id[:, :, 1][0, :]], '\n', ns[1, id[:, :, 1][1]])
        print(ns.shape[0])
        for i in range(ns.shape[0]):
            print("按第二列的数据大小进行排序:\n", ns[i, id[:, :, 1][i, :]])

        print("修改前的数组:\n", ns)
        ns[0, id[:, :, 1][0, 0]] = -10
        print("按第二列的数据大小进行排序,然后修改最小的一行:\n", ns)
        ns[0, id[:, :, 1][0, 0]][1] = -100
        print("按第二列的数据大小进行排序,然后修改最小的一行的第二个值:\n", ns)

        # print("id[:,:,1]:\n", id[:, :, 1])
        # print("按第二列的数据大小进行排序:\n", ns[id[:, 0]])

        # id = np.lexsort(ns[1])
        # print("id:\n", id)
        # print("ns排序后:\n", ns[:, id])

    def filter_test():
        ns = np.array([[[0, 0, 0, 0, 0, 100],
                        [1, 1, 1, 1, 1, 10000],
                        [5, 6, 6, 6, 6, 100],
                        [3, 3, 3, 3, 3, 100],
                        [4, 4, 4, 4, 4, 10000],
                        [2, 2, 2, 2, 2, 100]
                        ],
                       [[7, 0, 0, 0, 0, 10000],
                        [1, 1, 1, 1, 1, 100],
                        [5, 6, 6, 6, 6, 10000],
                        [3, 3, 3, 3, 3, 10000],
                        [0, 4, 4, 4, 4, 100],
                        [2, 2, 2, 2, 2, 100]
                        ]
                       ])
        print("原数据:\n", ns)

        # 筛选
        # 只计算三维数组的第一个二维数组
        id = np.zeros([ns.shape[1]], dtype=bool)
        for i in range(ns.shape[1]):
            id = id | (ns[0, :, 5] == 100)
        # print(id)
        print("筛选后数组:\n", ns[0, id])  # 之后把ns[0,id]看做一个整体;二维数组
        # 筛选后排序
        ns_filter = ns[0, id]
        print("ns_filter为:\n", ns_filter)

        id_sort = ns_filter.argsort(axis=0)
        print("id_sort为:\n", id_sort)
        print("按第一列排序后数组:\n", ns_filter[id_sort[:, 0]])
        # 排序后修改
        ns_filter[id_sort[:, 0][0]] = -1
        print("按第一列排序后数组,修改最小的一行为-1:\n", ns_filter[id_sort[:, 0]])
        ns_filter[id_sort[:, 0][0]][2] = 0
        print("按第一列排序后数组,修改最小的一行的第3个值为0:\n", ns_filter[id_sort[:, 0]])

        # 修改后返回原数据
        ns[0, id] = ns_filter
        print(ns)

    def test():
        # 计算三维数组的全部二维数组
        # 全部完整流程:
        ns = np.array([[[0, 0, 0, 0, 0, 100],
                        [1, 1, 1, 1, 1, 10000],
                        [5, 6, 6, 6, 6, 100],
                        [3, 3, 3, 3, 3, 100],
                        [4, 4, 4, 4, 4, 10000],
                        [2, 2, 2, 2, 2, 100]
                        ],
                       [[7, 0, 0, 0, 0, 10000],
                        [1, 1, 1, 1, 1, 100],
                        [5, 6, 6, 6, 6, 10000],
                        [3, 3, 3, 3, 3, 10000],
                        [0, 4, 4, 4, 4, 100],
                        [2, 2, 2, 2, 2, 100]
                        ]
                       ])
        print("原数据:\n", ns)

        for bs in range(ns.shape[0]):
            id = np.zeros([ns.shape[1]], dtype=bool)
            for i in range(ns.shape[1]):
                id = id | (ns[bs, :, 5] == 100)
            # print(id)
            # print("筛选后数组:\n", ns[0, id])  # 之后把ns[0,id]看做一个整体;二维数组
            # 筛选后排序
            ns_filter = ns[bs, id]
            # print("ns_filter为:\n", ns_filter)

            id_sort = ns_filter.argsort(axis=0)
            # print("id_sort为:\n", id_sort)
            # print("按第一列排序后数组:\n", ns_filter[id_sort[:, 0]])
            # 排序后修改
            ns_filter[id_sort[:, 0][0]] = -1
            # print("按第一列排序后数组,修改最小的一行为-1:\n", ns_filter[id_sort[:, 0]])
            ns_filter[id_sort[:, 0][0]][2] = 0
            # print("按第一列排序后数组,修改最小的一行的第3个值为0:\n", ns_filter[id_sort[:, 0]])

            # 修改后返回原数据
            ns[bs, id] = ns_filter

        print("操作完后的数据:\n", ns)


def text_write():
    path = os.path.join(os.getcwd(), "txt写入测试")
    with open(path, 'w') as f:
        f.write("hello! 这是txt写入测试.")

    ns = np.array(range(12)).reshape(3, 4)
    with open(path, 'w') as f:
        f.write("hello! 这是txt写入测试.\n" + ns.__str__())
        f.write("\nEND")


class Occ_filter():
    # 部分数据,测试足够用
    cls = np.array([[[7.004445, -2.940634, -2.2007353, -2.5283923],
                     [9.3398905, -3.9166932, -2.921009, -3.4019845],
                     [9.33281, -3.9009597, -2.9335585, -3.3912332],
                     [7.8270535, -2.1166255, -2.4785447, -3.5878234],
                     [7.465112, -1.6955433, -2.5043492, -3.5932987],
                     [5.415031, -0.83693886, -2.020754, -2.8435571]]])
    prob = np.array([[[0.99977916, 0.00004795, 0.0001005, 0.00007242],
                      [0.99999046, 0.00000175, 0.00000473, 0.00000293],
                      [0.9999906, 0.00000179, 0.00000471, 0.00000298],
                      [0.9999074, 0.00004803, 0.00003344, 0.00001103],
                      [0.9998323, 0.00010508, 0.0000468, 0.00001575],
                      [0.99723226, 0.00192132, 0.00058813, 0.00025831]]])
    # bbox_2d = np.array([[[0.07035355, -0.15154037, -0.5888689, 0.56354284],
    #                      [0.10060313, -0.2533346, -0.83627206, 0.7575691],
    #                      [0.10399755, -0.230512, -0.7829558, 0.754774],
    #                      [0.57526064, -0.2599153, 0.5070556, 0.66741943],
    #                      [0.7744001, -0.19142833, 0.6965968, 0.8053928],
    #                      [1.0210276, -0.07548057, 0.6491388, 0.7470195]]])
    bbox_2d = np.array([[[0.0, 0.0, 10.0, 10.0],
                         [9.0, 0.0, 10.0, 10.0],
                         [0.0, 25.0, 10.0, 10.0],
                         [10.0, 0.0, 10.0, 10.0],
                         [5.0, 25.0, 10.0, 10.0],
                         [10.0, 25.0, 10.0, 10.0]]])
    bbox_3d = np.array([[[-0.0502478, 0.05908747, - 0.90959895, 0.23877755, -0.6848674, 0.17785437],
                         [-0.09393481, 0.08450541, -1.2342116, 0.23911348, -0.89629656, 0.28122324],
                         [-0.07244383, 0.07180647, - 1.2367451, 0.24810761, -0.8726766, 0.26111692],
                         [-0.19424656, -0.753002, -0.07067724, 0.00846963, -0.8183509, 0.00125809],
                         [-0.07138748, -0.75582796, -0.1211952, 0.00485636, -0.84943235, -0.10937988],
                         [0.17544928, -0.57096535, -0.13709335, 0.0210136, -0.7015238, -0.15523729]]])
    threshold = 0.2  # 遮挡关系判定的阈值
    occ_correct = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.0])  # 这里先假定遮挡修正值
    cls = torch.from_numpy(cls).cuda()
    bbox_2d = torch.from_numpy(bbox_2d).cuda()
    bbox_3d = torch.from_numpy(bbox_3d).cuda()
    occ_correct = torch.from_numpy(occ_correct).cuda()
    print("bbox_3d.grad:", bbox_3d.grad)
    print("occ_correct.shape:", occ_correct.shape)
    print("occ_correct:", occ_correct)

    print("cls.shape:", cls.shape)
    print("torch.unsqueeze(cls).shap:", torch.unsqueeze(cls, 2).shape)


    # print(cls.shape[0])

    def Occlusion_v1(self, cls=None, bbox_2d=None, bbox_3d=None, threshold=0.3, occ_correct=None):
        # print(cls, cls.shape)
        # print(type(cls))
        cls = cls.cpu().detach().numpy()
        bbox_2d = bbox_2d.cpu().detach().numpy()
        bbox_3d = bbox_3d.cpu().detach().numpy()
        occ_correct = occ_correct.cpu().detach().numpy()
        # print(type(cls))
        # print("修改前的bbox_3d:\n", bbox_3d)

        for bs in range(cls.shape[0]):
            # cls转化为类型编码,并由类型编码筛选出类型为car的数据的索引;
            typeEncode = np.argmax(cls[bs, :, :], axis=1) + 1
            idx_filter = np.zeros([cls.shape[1]], dtype=bool)
            for i in range(cls.shape[1]):
                idx_filter |= typeEncode == 1

            # 由索引过滤出类型为car的数据
            bbox2d_filter = bbox_2d[bs, idx_filter]
            bbox3d_filter = bbox_3d[bs, idx_filter]
            # print("bbox3d_filter修正前:\n", bbox3d_filter)
            # x,y,w,h --> x1,y1,x2,y2
            bbox2d_filter = bbXYWH2Coords(bbox2d_filter)

            # 筛选后按深度大小排序
            idx_sort = bbox3d_filter.argsort(axis=0)[:, 2]
            print("id_sort:\n", idx_sort)

            # 计算iou,判定遮挡关系,进行遮挡修正
            for count in range(0, idx_sort.shape[0] - 1):
                # print(count)
                # threshold = 0.2         # 遮挡关系判定的阈值
                # occ_correct = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.0])   # 这里先假定遮挡修正值
                iouValue = np.zeros(bbox2d_filter.shape[0])  # 初始化全部为0;
                idx_occlusion = np.zeros(bbox2d_filter.shape[0], dtype=bool)  # 初始化全部为False;

                iouValue[count + 1:] = iou(bbox2d_filter[idx_sort[count:count + 1]],
                                           bbox2d_filter[idx_sort[count + 1:]])
                # print("iouValue:", iouValue)
                for occid in range(count + 1, idx_occlusion.shape[0]):
                    # print(occid)
                    idx_occlusion[occid] |= (iouValue[occid] > threshold)
                # print("遮挡关系:", idx_occlusion)
                # print("当前值为:", bbox3d_filter[idx_sort[count]])
                bbox3d_filter[idx_sort[idx_occlusion]] += bbox3d_filter[idx_sort[
                    count]] * occ_correct  # 用'='表示返回的是修正值,用'+='返回的是修改后的值;
                # print("bbox3d_filter修正后:\n", bbox3d_filter)

            bbox2d_filter = bbCoords2XYWH(bbox2d_filter)  # 变回x,y,w,h模式,其实这一步不太需要,因为从始至终都没改变bbox_2d的值;

            # 将修改完的值返回给原数据
            bbox_3d[bs, idx_filter] = bbox3d_filter

        # print("修改后的bbox_3d:\n", bbox_3d)

        bbox_3d = torch.from_numpy(bbox_3d).cuda()
        return bbox_3d
    '''v1版本中将数据都转化为ndarray进行计算,由于计算量较大且只用到cpu,导致计算时间过长;v2版本旨在解决此问题;
       v2版本不再将源数据转为ndarray进行计算,全部使用torch.Tenser进行计算,并根据Tenser的特点改变了部分代码实现.
       参考了:'''
    def Occlusion_v2(self, cls=None, bbox_2d=None, bbox_3d=None, threshold=0.3, occ_correct=None):
        # print(cls, cls.shape)
        # print(type(cls))
        # print(type(cls))
        print("修改前的bbox_3d:\n", bbox_3d)

        for bs in range(cls.shape[0]):
            # cls转化为类型编码,并由类型编码筛选出类型为car的数据的索引;
            typeEncode = torch.argmax(cls[bs, :, :], 1) + 1
            # print("typeEncode:", typeEncode)
            # idx_zeros = torch.zeros(cls.shape[1]).cuda().byte()
            # idx_ones = torch.ones(cls.shape[1]).cuda().byte()
            # print("idx_ones", idx_ones)
            idx_filter = torch.where(typeEncode == 1, torch.ones(1).cuda().byte(), torch.zeros(1).cuda().byte())
            print("idx_filter:", idx_filter)

            # 由索引过滤出类型为car的数据
            bbox2d_filter = bbox_2d[bs, idx_filter]
            bbox3d_filter = bbox_3d[bs, idx_filter]
            # print("bbox3d_filter修正前:\n", bbox3d_filter)
            # x,y,w,h --> x1,y1,x2,y2
            bbox2d_filter = bbXYWH2Coords(bbox2d_filter)
            # print("bbox2d_filter的type: {}".format(type(bbox2d_filter)))

            # 筛选后按深度大小排序
            idx_sort = bbox3d_filter.argsort(0)[:, 2]
            # print("id_sort:\n", idx_sort)

            # 计算iou,判定遮挡关系,进行遮挡修正
            for count in range(0, idx_sort.shape[0] - 1):
                print("第 {} 次循环:".format(count))
                # threshold = 0.2         # 遮挡关系判定的阈值
                # occ_correct = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.0])   # 这里先假定遮挡修正值
                iouValue = torch.zeros(bbox2d_filter.shape[0]).cuda()  # 初始化全部为0;
                # idx_occlusion = torch.zeros(bbox2d_filter.shape[0]).cuda().byte()  # 初始化全部为False;

                # iouValue[count + 1:] = torch.from_numpy(iou(bbox2d_filter[idx_sort[count:count + 1]].cpu().detach().numpy(),
                #                            bbox2d_filter[idx_sort[count + 1:]].cpu().detach().numpy()))
                iouValue[count + 1:] = iou(bbox2d_filter[idx_sort[count:count + 1]], bbox2d_filter[idx_sort[count + 1:]])
                print("iouValue:", iouValue)
                idx_occlusion = torch.where(iouValue > threshold, torch.ones(1).cuda().byte(), torch.zeros(1).cuda().byte())
                # for occid in range(count + 1, idx_occlusion.shape[0]):
                #     # print(occid)
                #     idx_occlusion[occid] |= (iouValue[occid] > threshold)
                print("遮挡关系:", idx_occlusion)
                print("当前值为:", bbox3d_filter[idx_sort[count]])
                bbox3d_filter[idx_sort[idx_occlusion]] += bbox3d_filter[idx_sort[
                    count]] * occ_correct  # 用'='表示返回的是修正值,用'+='返回的是修改后的值;
                # print("bbox3d_filter修正后:\n", bbox3d_filter)

            bbox2d_filter = bbCoords2XYWH(bbox2d_filter)  # 变回x,y,w,h模式,其实这一步不太需要,因为从始至终都没改变bbox_2d的值;

            # 将修改完的值返回给原数据
            bbox_3d[bs, idx_filter] = bbox3d_filter

        print("修改后的bbox_3d:\n", bbox_3d)

        return bbox_3d
    '''v2版本中出现梯度问题，原因在于直接在原数据上进行索引、分割、修改，导致在后向传播中产生梯度问题；v3版本旨在解决此问题;
       v3版本计划采用两个方法来解决：detach()方法、不再在原数据上直接修改而是采用增量相加；
       参考了：rpn_3d.py中数据复制和新增数据的处理方式'''
    def Occlusion_v3(self, cls=None, bbox_2d=None, bbox_3d=None, threshold=0.3, occ_correct=None):
        # print(cls, cls.shape)
        # print(type(cls))
        # print(type(cls))
        print("修改前的bbox_3d:\n", bbox_3d)

        bbox_3d_correct = torch.zeros_like(bbox_3d)
        bbox_3d_correct = torch.tensor(bbox_3d_correct, requires_grad=False)

        for bs in range(cls.shape[0]):
            # ======cls转化为类型编码,并由类型编码筛选出类型为car的数据的索引;
            typeEncode = torch.argmax(cls[bs, :, :], 1) + 1
            # print("typeEncode:", typeEncode)
            idx_filter = torch.where(typeEncode == 1, torch.ones(1).cuda().byte(), torch.zeros(1).cuda().byte())
            print("idx_filter:", idx_filter)

            # ======由索引过滤出类型为car的数据
            bbox2d_filter = bbox_2d[bs, idx_filter]
            bbox2d_filter = torch.tensor(bbox2d_filter, requires_grad=False)  # .type(torch.FloatTensor).cuda()
            bbox3d_filter = bbox_3d[bs, idx_filter]
            print("bbox3d_filter.grad:", bbox3d_filter.grad)
            bbox3d_filter = torch.tensor(bbox3d_filter, requires_grad=False)  # .type(torch.FloatTensor).cuda()
            print("bbox3d_filter.grad:", bbox3d_filter.grad)
            bbox3d_filter_tmp = bbox3d_filter.clone()
            # print("bbox3d_filter修正前:\n", bbox3d_filter)
            # x,y,w,h --> x1,y1,x2,y2
            bbox2d_filter = bbXYWH2Coords(bbox2d_filter)
            # print("bbox2d_filter的type: {}".format(type(bbox2d_filter)))

            # ======筛选后按深度大小排序
            idx_sort = bbox3d_filter.argsort(0)[:, 2]
            # print("id_sort:\n", idx_sort)

            # ======计算iou,判定遮挡关系,进行遮挡修正
            for count in range(0, idx_sort.shape[0] - 1):
                print("第 {} 次循环:".format(count))

                iouValue = torch.zeros(bbox2d_filter.shape[0]).cuda()  # 初始化全部为0;

                iouValue[count + 1:] = iou(bbox2d_filter[idx_sort[count:count + 1]], bbox2d_filter[idx_sort[count + 1:]])
                print("iouValue:", iouValue)

                idx_occlusion = torch.where(iouValue > threshold, torch.ones(1).cuda().byte(), torch.zeros(1).cuda().byte())
                print("遮挡关系:", idx_occlusion)
                print("当前值为:", bbox3d_filter[idx_sort[count]])

                # 用'='表示返回的是修正值,用'+='返回的是修改后的值;
                bbox3d_filter[idx_sort[idx_occlusion]] = bbox3d_filter[idx_sort[idx_occlusion]] + bbox3d_filter[idx_sort[count]] * occ_correct
                # print("bbox3d_filter修正后:\n", bbox3d_filter)

            bbox2d_filter = bbCoords2XYWH(bbox2d_filter)  # 变回x,y,w,h模式,其实这一步可能不太需要,因为从始至终都没改变bbox_2d的值;

            bbox_3d_correct[bs, idx_filter] = bbox3d_filter - bbox3d_filter_tmp
        print("bbox_3d_correct:", bbox_3d_correct)
        # ======将修改完的值返回给原数据
        bbox_3d = bbox_3d + bbox_3d_correct
        print("修改后的bbox_3d:\n", bbox_3d)

        return bbox_3d
    '''V4版本对输入数据的要求发生了变化，去掉了batch维度，数据由三维变为二维，相应地，去掉了遍历batch的循环，
        但inplace操作导致的错误仍未解决'''
    def Occlusion_v4(self, cls=None, bbox_2d=None, bbox_3d=None, threshold=0.3, occ_correct=None):
        '''
        cls: tensor.cuda, ( N, 4),
        bbox_2d: tensor.cuda, ( N, 4),
        bbox_3d: tensor.cuda, ( N, 7),
        occ_correct: tensor.cuda, (7),
        threshold: float in 0~1,
        '''
        # print("修改前的bbox_3d:\n", bbox_3d)
        # bbox_3d_correct = torch.zeros_like(bbox_3d)
        # bbox_3d_correct = torch.tensor(bbox_3d_correct, requires_grad=False)
        # occ_correct = occ_correct.detach()


        # cls转化为类型编码,并由类型编码筛选出类型为car的数据的索引;
        typeEncode = torch.argmax(cls[:, :], 1) + 1
        # print("typeEncode:", typeEncode)
        idx_filter = torch.where(typeEncode == 1, torch.ones(1).cuda().byte(), torch.zeros(1).cuda().byte())
        idx_filter = idx_filter.bool()
        # print("idx_filter:", idx_filter)

        # 由索引过滤出类型为car的数据
        bbox2d_filter = bbox_2d[idx_filter]
        # bbox2d_filter = torch.tensor(bbox2d_filter, requires_grad=False)
        bbox3d_filter = bbox_3d[idx_filter]
        bbox3d_filter = bbox3d_filter.detach()
        # bbox3d_filter_tmp = bbox3d_filter.clone()
        # print("bbox3d_filter修正前:\n", bbox3d_filter)
        # x,y,w,h --> x1,y1,x2,y2
        bbox2d_filter = bbXYWH2Coords(bbox2d_filter)
        # print("bbox2d_filter的type: {}".format(type(bbox2d_filter)))

        # 筛选后按深度大小排序
        idx_sort = bbox3d_filter.argsort(0)[:, 2]
        # print("id_sort:\n", idx_sort)

        # 计算iou,判定遮挡关系,进行遮挡修正
        time_start = time()
        for count in range(0, idx_sort.shape[0] - 1):
            # print("第 {} 次循环:".format(count))
            iouValue = torch.zeros(bbox2d_filter.shape[0]).cuda()  # 初始化全部为0;

            iouValue[count + 1:] = iou(bbox2d_filter[idx_sort[count:count + 1]],
                                       bbox2d_filter[idx_sort[count + 1:]])
            # print("iouValue:", iouValue)
            idx_occlusion = torch.where(iouValue > threshold, torch.ones(1).cuda().byte(),
                                        torch.zeros(1).cuda().byte())
            # print("遮挡关系:", idx_occlusion)
            # print("当前值为:", bbox3d_filter[idx_sort[count]])
            # print("occ_correct:", occ_correct)
            # print("bbox3d_filter.grad_fn:", bbox3d_filter.grad_fn)
            correct_value = bbox3d_filter[idx_sort[count]].detach() * occ_correct
            # print("correct_value:", correct_value)
            bbox3d_filter[idx_sort[idx_occlusion]] = bbox3d_filter[idx_sort[
                idx_occlusion]] + correct_value  # 用'='表示返回的是修正值,用'+='返回的是修改后的值;
            # print("bbox3d_filter[idx_sort[idx_occlusion]]:", bbox3d_filter[idx_sort[idx_occlusion]])
            # print("bbox3d_filter修正后:\n", bbox3d_filter)

        bbox2d_filter = bbCoords2XYWH(bbox2d_filter)  # 变回x,y,w,h模式,其实这一步不太需要,因为从始至终都没改变bbox_2d的值;
        print("用了{}秒".format(time() - time_start))
        # 将修改完的值返回给原数据
        # bbox_3d_correct[bs, idx_filter] = bbox3d_filter - bbox3d_filter_tmp
        bbox_3d[idx_filter] = bbox3d_filter

        # bbox_3d = bbox_3d + bbox_3d_correct

        # print("修改后的bbox_3d:\n", bbox_3d)
        return bbox_3d
    '''V5版本解决了inplace操作导致的错误，但仍需优化， 前几个版本的遗留的现版本不需要的操作也需清理'''
    def Occlusion_v5(self, cls=None, bbox_2d=None, bbox_3d=None, threshold=0.3, occ_correct=None):
        '''
        cls: tensor.cuda, (N, 4),
        bbox_2d: tensor.cuda, (N, 4),
        bbox_3d: tensor.cuda, (N, 7),
        occ_correct: tensor.cuda, (7),
        threshold: float in 0~1,
        '''
        print("修改前的bbox_3d:\n", bbox_3d)

        # cls转化为类型编码,并由类型编码筛选出类型为car的数据的索引;
        typeEncode = torch.argmax(cls[:, :], 1) + 1
        # print("typeEncode:", typeEncode)
        tensor_ones = torch.ones(1).cuda().bool()
        tensor_zeros = torch.zeros(1).cuda().bool()
        idx_filter = torch.where(typeEncode == 1, tensor_ones, tensor_zeros)
        print("idx_filter:", idx_filter)

        # 由索引过滤出类型为car的数据
        bbox2d_filter = bbox_2d[idx_filter]
        bbox3d_filter = bbox_3d[idx_filter]
        bbox3d_filter = bbox3d_filter.detach()
        bbox3d_stack = torch.zeros((bbox3d_filter.shape[0], bbox3d_filter.shape[0], bbox3d_filter.shape[1])).type(torch.DoubleTensor).cuda()
        if bbox3d_filter.shape[0] > 0:
            bbox3d_stack[0] = bbox3d_filter
        # bbox3d_filter_tmp = bbox3d_filter.clone()
        # print("bbox3d_filter修正前:\n", bbox3d_filter)
        # x,y,w,h --> x1,y1,x2,y2
        bbox2d_filter = bbXYWH2Coords(bbox2d_filter)
        # print("bbox2d_filter的type: {}".format(type(bbox2d_filter)))

        # 筛选后按深度大小排序
        idx_sort = bbox3d_filter.argsort(0)[:, 2]
        # print("id_sort:\n", idx_sort)

        print("car_num_now: {}".format(idx_sort.shape[0]))
        # self.car_num_max = idx_sort.shape[0] if idx_sort.shape[0] > self.car_num_max else self.car_num_max
        # self.car_num_min = idx_sort.shape[0] if idx_sort.shape[0] < self.car_num_min else self.car_num_min
        # print("car_num_max: {} , car_num_min: {} ".format(self.car_num_max, self.car_num_min))
        # car_num_path = os.path.join(os.getcwd(), "output", "record", "car_num数记录_{}".format(self.now))
        # with open(car_num_path, 'a') as f:
        #     f.write("{}\n".format(idx_sort.shape[0]))

        # print("occ_correct.shape:", occ_correct.shape)
        # print("occ_correct:", occ_correct)

        # 计算iou,判定遮挡关系,进行遮挡修正
        time_start = time()
        for count in range(0, idx_sort.shape[0] - 1):
            print("第 {} 次循环:".format(count))
            iouValue = torch.zeros(bbox2d_filter.shape[0]).cuda()  # 初始化全部为0;

            iouValue[count + 1:] = iou(bbox2d_filter[idx_sort[count:count + 1]], bbox2d_filter[idx_sort[count + 1:]])
            print("iouValue:", iouValue)

            idx_occlusion = torch.where(iouValue > threshold, torch.ones(1).cuda().byte(),
                                        torch.zeros(1).cuda().byte())
            print("遮挡关系:", idx_occlusion)
            print("当前值为:", bbox3d_filter[idx_sort[count]])
            # print("occ_correct:", occ_correct)
            correct_value = bbox3d_filter[idx_sort[count]].detach() * occ_correct
            # print("correct_value:", correct_value)
            bbox3d_stack[count + 1] = bbox3d_stack[count]
            bbox3d_stack[count + 1, idx_sort[idx_occlusion]] = bbox3d_stack[count, idx_sort[idx_occlusion]] + correct_value
            # print("bbox3d_filter[idx_sort[idx_occlusion]]:", bbox3d_filter[idx_sort[idx_occlusion]])
            # print("bbox3d_filter修正后:\n", bbox3d_filter)

        bbox2d_filter = bbCoords2XYWH(bbox2d_filter)  # 变回x,y,w,h模式,其实这一步不太需要,因为从始至终都没改变bbox_2d的值;
        # print("用了{}秒".format(time() - time_start))
        # 将修改完的值返回给原数据
        print("bbox3d_stack:\n", bbox3d_stack)
        bbox_3d[idx_filter] = bbox3d_stack[-1]

        print("修改后的bbox_3d:\n", bbox_3d)
        return bbox_3d

if __name__ == "__main__":
    # iou_test()
    # size()
    # anchors()
    # pkl_read()
    # ModulePrint()
    # nptest()
    # image_read()

    # Filtrate.modify_test(Filtrate.testset)
    # Filtrate.filter_test()
    # Filtrate.test()

    # text_write()

    # Occ_filter().Occlusion_v1(Occ_filter.cls, Occ_filter.bbox_2d, Occ_filter.bbox_3d, Occ_filter.threshold, Occ_filter.occ_correct)
    # print("最后的bbox_3d:\n", bbox_3d)
    # Occ_filter().Occlusion_v2(Occ_filter.cls, Occ_filter.bbox_2d, Occ_filter.bbox_3d, Occ_filter.threshold, Occ_filter.occ_correct)
    # print("最后的bbox_3d:\n", bbox_3d)
    # Occ_filter().Occlusion_v3(Occ_filter.cls, Occ_filter.bbox_2d, Occ_filter.bbox_3d, Occ_filter.threshold,Occ_filter.occ_correct)
    Occ_filter().Occlusion_v5(Occ_filter.cls[0], Occ_filter.bbox_2d[0], Occ_filter.bbox_3d[0], Occ_filter.threshold, Occ_filter.occ_correct)