# -----------------------------------------
# python modules
# -----------------------------------------
from easydict import EasyDict as edict
from getopt import getopt
import numpy as np
import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.core import *
from lib.imdb_util import *
from lib.loss.rpn_3d import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.autograd.set_detect_anomaly(True)

def main(argv):

    # -----------------------------------------
    # parse arguments # 读取参数列表进行解析
    # -----------------------------------------
    opts, args = getopt(argv, '', ['config=', 'restore='])

    # defaults
    conf_name = None
    restore = None

    # read opts
    for opt, arg in opts:

        if opt in ('--config'): conf_name = arg         # 配置文件名称，为depth_guided_config
        if opt in ('--restore'): restore = int(arg)     # 训练中断后继续训练，为重开始次数。

    # required opt
    if conf_name is None:
        raise ValueError('Please provide a configuration file name, e.g., --config=<config_name>')

    # -----------------------------------------
    # basic setup # 加载配置文件，初始化相关文件及路径、可视化设置、一些其他设置。
    # -----------------------------------------

    conf = init_config(conf_name)  # 加载配置文件(模型)（scripts.config.depth_guided_config）; conf类型为字典;

    if restore:     # 原程序没有；用于中断训练后的恢复训练。
        ''' 
        在./output/depth_guided_config/目录下建立软连接,连接到要继续的结果目录，并将软连接目录名变为restore_dir; 其中结果目录名
        类似:Adaptive_block2_resnet_dilate50_batch1_dropoutearly0_5_lr0_01_onecycle_iter40000_2021_01_05_16_36_29
        '''
        conf.result_dir = "restore_dir"

    paths = init_training_paths(conf_name, conf.result_dir)  # 产生相关路径字典，创建相关文件夹. output;weights;logs;results
    init_torch(conf.rng_seed, conf.cuda_seed)  # 设置随机种子、使代码特定性，用以保证实验的可重复性，使得多次运行的结果完全一致
    init_log_file(paths.logs)  # 在./output/depth_guided_config/<result_dir>/log文件夹下写log文件，以格式化的时间戳为文件名

    vis = init_visdom(conf.result_dir, conf.visdom_port)  # 初始化一个visdom会话;如果没有运行visdom服务器(外部)，函数将返回'None'。

    # defaults
    start_iter = 0
    tracker = edict()
    iterator = None
    has_visdom = vis is not None

    dataset = Dataset(conf, paths.data, paths.output)   # 数据集的格式化处理
    generate_anchors(conf, dataset.imdb, paths.output)  # 生成anchors
    compute_bbox_stats(conf, dataset.imdb, paths.output) # 计算由生成的anchors确定的所有rois相对于对应ground truth的相对误差的均值与方差

    paths.output = os.path.join(paths.output, conf.result_dir)  # 注意:这里变更了output指代的路径

    # -----------------------------------------
    # store config
    # -----------------------------------------

    # store configuration   生成配置文件conf.pkl;
    # 相较于depth_guided_config.py,有变化:generate_anchors中更新了conf.anchors; compute_bbox_stat更新了conf.bbox_means、conf.bbox_stds;
    pickle_write(os.path.join(paths.output, 'conf.pkl'), conf)

    # show configuration
    pretty = pretty_print('conf', conf)
    logging.info(pretty)


    # -----------------------------------------
    # network and loss
    # -----------------------------------------
# readhere 0
    # training network
    rpn_net, optimizer, scheduler = init_training_model(conf, paths.output, conf_name)

    # setup loss
    criterion_det = RPN_3D_loss(conf)

    # custom pretrained network
    if 'pretrained' in conf:

        load_weights(rpn_net, conf.pretrained)

    # resume training
    if restore:
        start_iter = (restore - 1)
        resume_checkpoint(optimizer, rpn_net, paths.weights, restore)

    freeze_blacklist = None if 'freeze_blacklist' not in conf else conf.freeze_blacklist    # None
    freeze_whitelist = None if 'freeze_whitelist' not in conf else conf.freeze_whitelist    # None

    freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist)

    optimizer.zero_grad()

    start_time = time()
    # -----------------------------------------
    # train
    # -----------------------------------------

    for iteration in range(start_iter, conf.max_iter):

        # next iteration
        iterator, images, depths, imobjs = next_iteration(dataset.loader, iterator)

        print("======images.size:", images.size())
        #  learning rate
        adjust_lr(conf, optimizer, iteration, scheduler)

        # forward
        if conf.corner_in_3d:
            cls, prob, bbox_2d, bbox_3d, feat_size, bbox_vertices, corners_3d = rpn_net(images.cuda(), depths.cuda())
        elif conf.use_corner:
            cls, prob, bbox_2d, bbox_3d, feat_size, bbox_vertices = rpn_net(images.cuda(), depths.cuda())
        else:
            cls, prob, bbox_2d, bbox_3d, feat_size = rpn_net(images.cuda(), depths.cuda())
        # print('cls:{}, {}\n prob:{}, {}\n bbox_2d:{}, {}\n bbox_3d:{}, {}\n'.format(cls, cls.size(), prob, prob.size(), bbox_2d, bbox_2d.size(), bbox_3d, bbox_3d.size()))
        feat_size = feat_size[:2]  # feat_size = [32, 110]
        # print(feat_size)
        # loss
        if conf.corner_in_3d:
            det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size, bbox_vertices, corners_3d)
        elif conf.use_corner:
            det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size, bbox_vertices)
        else:
            det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size)

        total_loss = det_loss
        stats = det_stats

        # backprop
        if total_loss > 0:

            total_loss.backward()

            # batch skip, simulates larger batches by skipping gradient step
            if (not 'batch_skip' in conf) or ((iteration + 1) % conf.batch_skip) == 0:
                optimizer.step()
                optimizer.zero_grad()   # 这里将优化器的梯度归零；

        # keep track of stats
        compute_stats(tracker, stats)

        # -----------------------------------------
        # display
        # -----------------------------------------
        if (iteration + 1) % conf.display == 0 and iteration > start_iter:  # 每（conf.display=）50次迭代打印一次

            # log results
            log_stats(tracker, iteration, start_time, start_iter, conf.max_iter)

            # display results
            if has_visdom:
                display_stats(vis, tracker, iteration, start_time, start_iter, conf.max_iter, conf_name, pretty)

            # reset tracker
            tracker = edict()

        # -----------------------------------------
        # test network
        # -----------------------------------------
        if (iteration + 1) % conf.snapshot_iter == 0 and iteration > start_iter:  # 每（conf.snapshot_iter =）5000次迭代进项一次测试

            # store checkpoint
            save_checkpoint(optimizer, rpn_net, paths.weights, (iteration + 1))

            if conf.do_test:        # True
                rpn_net = rpn_net.module
                # eval mode
                rpn_net.eval()

                # necessary paths
                results_path = os.path.join(paths.results, 'results_{}'.format((iteration + 1)))

                # -----------------------------------------
                # test kitti
                # -----------------------------------------
                if conf.test_protocol.lower() == 'kitti':

                    # delete and re-make
                    results_path = os.path.join(results_path, 'data')
                    mkdir_if_missing(results_path, delete_if_exist=True)

                    test_kitti_3d(conf.dataset_test, 'validation', rpn_net, conf, results_path, paths.data)

                else:
                    logging.warning('Testing protocol {} not understood.'.format(conf.test_protocol))

                # train mode
                rpn_net = torch.nn.DataParallel(rpn_net)
                rpn_net.train()

                freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist)

    print(conf.result_dir)


# run from command line
if __name__ == "__main__":
    main(sys.argv[1:])
