import os
import pickle

# 1. 在列表中加入 'sop'
DATASETS = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k', 'sop']

def configdataset(dataset, dir_main):

    dataset = dataset.lower()

    if dataset not in DATASETS:    
        raise ValueError('Unknown dataset: {}!'.format(dataset))

    # loading imlist, qimlist, and gnd, in cfg as a dict
    gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
    with open(gnd_fname, 'rb') as f:
        cfg = pickle.load(f)
    cfg['gnd_fname'] = gnd_fname

    cfg['ext'] = '.jpg'
    cfg['qext'] = '.jpg'

    # 2. 修改图片路径逻辑
    if dataset == 'sop':
        # SOP 的特殊处理：图片还在 data/train/sop/ims 里
        cfg['dir_data'] = os.path.join(dir_main, dataset)
        # dir_main 通常是 'data/test'，所以 dirname 是 'data'
        cfg['dir_images'] = os.path.join(os.path.dirname(dir_main), 'train', 'sop', 'ims')
    else:
        # 其他数据集的默认逻辑
        cfg['dir_data'] = os.path.join(dir_main, dataset)
        cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')

    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])

    cfg['im_fname'] = config_imname
    cfg['qim_fname'] = config_qimname

    cfg['dataset'] = dataset

    return cfg

def config_imname(cfg, i):
    # 修改：如果是 sop，列表里自带后缀，直接用
    if cfg['dataset'] == 'sop':
        return os.path.join(cfg['dir_images'], cfg['imlist'][i])
    else:
        return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])

def config_qimname(cfg, i):
    # 修改：如果是 sop，列表里自带后缀，直接用
    if cfg['dataset'] == 'sop':
        return os.path.join(cfg['dir_images'], cfg['qimlist'][i])
    else:
        return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])