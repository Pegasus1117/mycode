from easydict import EasyDict as edict
import numpy as np
import datetime

def Config():

    conf = edict()
        
    # ----------------------------------------
    #  general
    # ----------------------------------------

    conf.model = 'resnet_dilate'
    conf.lr = 0.01            # 学习率
    conf.max_iter = 40000     # 最大迭代数
    conf.use_dropout = True
    conf.drop_channel = True
    conf.dropout_rate = 0.5
    conf.dropout_position = 'early'  # 'early'  'late' 'adaptive'
    conf.do_test = True
    conf.lr_policy = 'onecycle'  # 'onecycle'  # 'cosinePoly'  # 'cosineRestart'  # 'poly'
    conf.restart_iters = 5000
    conf.batch_size = 1
    conf.base_model = 50
    conf.depth_channel = 1
    conf.adaptive_diated = True
    conf.use_seg = False
    conf.use_corner = False
    conf.corner_in_3d = False
    conf.use_hill_loss = False
    conf.use_rcnn_pretrain = False
    conf.deformable = False

    conf.alias = 'Adaptive_block2'

    conf.result_dir = '_'.join([conf.alias, conf.model + str(conf.base_model), 'batch' + str(conf.batch_size),
                                'dropout' + conf.dropout_position + str(conf.dropout_rate), 'lr' + str(conf.lr),
                                conf.lr_policy, 'iter' + str(conf.max_iter),
                                datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")]).replace('.', '_').replace(':', '_').replace('-', '_')


    # solver settings
    conf.solver_type = 'sgd'     # 随机梯度下降

    conf.momentum = 0.9          # 超参数：动（冲）量：动量越大时，其转换为势能的能量也就越大，就越有可能摆脱局部凹域的束缚（跳过局部最优点），进入全局凹域。常设为0.9
    conf.weight_decay = 0.0005   # 权重衰减：防止过拟合

    conf.snapshot_iter = 5000
    conf.display = 50

    
    # sgd parameters

    conf.lr_steps = None
    conf.lr_target = conf.lr * 0.00001
    
    # random   随机种子，用于保证实验的可重复性
    conf.rng_seed = 2
    conf.cuda_seed = 2
    
    # misc network
    conf.image_means = [0.485, 0.456, 0.406]      # 图像的均值，用于在数据增强中进行归一化
    conf.image_stds = [0.229, 0.224, 0.225]       # 图像的方差，用于在数据增强中进行归一化
    if conf.use_rcnn_pretrain:
        conf.image_means = [102.9801, 115.9465, 122.7717]  # conf.image_means[::-1]
        conf.image_stds = [1, 1, 1]  # conf.image_stds[::-1]
    if conf.use_seg:
        conf.depth_mean = [4413.160626995486, 4413.160626995486, 5.426258330316642]    # 深度图像的均值，用于在数据增强中进行归一化
        conf.depth_std = [3270.0158918863494, 3270.0158918863494, 0.5365540402943388]  # 深度图像的方差，用于在数据增强中进行归一化
    else:
        conf.depth_mean = [4413.160626995486, 4413.160626995486, 4413.160626995486]  # DORN
        conf.depth_std = [3270.0158918863494, 3270.0158918863494, 3270.0158918863494]
        # conf.depth_mean = [8295.013626842678, 8295.013626842678, 8295.013626842678]  # PSM
        # conf.depth_std = [5134.9781439128665, 5134.9781439128665, 5134.9781439128665]
        # conf.depth_mean = [30.83664619525601, 30.83664619525601, 30.83664619525601]  # DISP
        # conf.depth_std = [19.992999492848206, 19.992999492848206, 19.992999492848206]
    if conf.depth_channel == 3:
        conf.depth_mean = [137.39162828, 40.58310471, 140.70854621]  # MONO1
        conf.depth_std = [33.75859339, 51.479677, 65.254889]
        conf.depth_mean = [107.0805491, 68.26778312, 133.50751215]  # MONO2
        conf.depth_std = [38.65614623, 73.59464917, 88.24401221]

    conf.feat_stride = 16
    
    conf.has_3d = True

    # ----------------------------------------
    #  image sampling and datasets
    # ----------------------------------------

    # scale sampling  
    conf.test_scale = 512
    conf.crop_size = [512, 1760]   # 裁剪大小
    conf.mirror_prob = 0.50        # 水平翻转概率
    conf.distort_prob = -1         # 扭曲（变形）概率
    
    # datasets
    conf.dataset_test = 'kitti_split1'
    conf.datasets_train = [{'name': 'kitti_split1', 'anno_fmt': 'kitti_det', 'im_ext': '.png', 'scale': 1}]
    conf.use_3d_for_2d = True
    
    # percent expected height ranges based on test_scale
    # used for anchor selection 
    conf.percent_anc_h = [0.0625, 0.75]
    
    # labels settings
    conf.min_gt_h = conf.test_scale*conf.percent_anc_h[0]   # =512*0.0625=32.0
    conf.max_gt_h = conf.test_scale*conf.percent_anc_h[1]   # =512*0.75=384.0
    conf.min_gt_vis = 0.65
    conf.ilbls = ['Van', 'ignore']
    conf.lbls = ['Car', 'Pedestrian', 'Cyclist']
    
    # ----------------------------------------
    #  detection sampling
    # ----------------------------------------
    
    # detection sampling

    conf.fg_image_ratio = 1.0       # 前景(foreground)图像比率，用于平衡采样中
    conf.box_samples = 0.20
    conf.fg_fraction = 0.20
    conf.bg_thresh_lo = 0
    conf.bg_thresh_hi = 0.5
    conf.fg_thresh = 0.5
    conf.ign_thresh = 0.5
    conf.best_thresh = 0.35

    # ----------------------------------------
    #  inference and testing
    # ----------------------------------------

    # nms
    conf.nms_topN_pre = 3000
    conf.nms_topN_post = 40
    conf.nms_thres = 0.4
    conf.clip_boxes = False

    conf.test_protocol = 'kitti'
    conf.test_db = 'kitti'
    conf.test_min_h = 0
    conf.min_det_scales = [0, 0]

    # ----------------------------------------
    #  anchor settings
    # ----------------------------------------
    
    # clustering settings
    conf.cluster_anchors = 0
    conf.even_anchors = 0
    conf.expand_anchors = 0
                             
    conf.anchors = None

    conf.bbox_means = None
    conf.bbox_stds = None
    
    # initialize anchors     # 论文中有说明；
    base = (conf.max_gt_h / conf.min_gt_h) ** (1 / (12 - 1))
    conf.anchor_scales = np.array([conf.min_gt_h * (base ** i) for i in range(0, 12)])
    conf.anchor_ratios = np.array([0.5, 1.0, 1.5])
    
    # loss logic
    conf.hard_negatives = True
    conf.focal_loss = 1
    conf.cls_2d_lambda = 1
    conf.iou_2d_lambda = 0
    conf.bbox_2d_lambda = 1
    conf.bbox_3d_lambda = 1
    conf.bbox_3d_proj_lambda = 0.0
    
    conf.hill_climbing = True
    
    # visdom
    conf.visdom_port = 9891

    return conf

