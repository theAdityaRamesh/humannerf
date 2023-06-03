import os
import imp
import time

import numpy as np
import torch

from core.utils.file_util import list_files
from configs import cfg
from .dataset_args import DatasetArgs


def _query_dataset(data_type):
    # get the dataset module for the datatype
    # from configs.human_nerf.zju_mocap.{sub_name}.adventure.yaml
    module = cfg[data_type].dataset_module
    # replace . with / to create path and 
    # append .py
    module_path = module.replace(".", "/") + ".py"
    # import the module from module_path
    # eg :
    #   module_path : core/data.human_nerf/freeview.py
    #   module : core.data.human_nerf.freeview
    dataset = imp.load_source(module, module_path).Dataset
    return dataset


def _get_total_train_imgs(dataset_path):
    # go to dataset_path/images
    # find files ending with .png
    # return its length
    train_img_paths = \
        list_files(os.path.join(dataset_path, 'images'),
                                exts=['.png'])
    return len(train_img_paths)


def create_dataset(data_type='train'):
    # get dataset name from adventure.yaml
    dataset_name = cfg[data_type].dataset
    # get dataset args corresponding to dataset name
    # common to all data_types : train/freeview/movement
    args = DatasetArgs.get(dataset_name)
    # customize dataset arguments according to dataset type
    args['bgcolor'] = None if data_type == 'train' else cfg.bgcolor
    if data_type == 'progress':
        total_train_imgs = _get_total_train_imgs(args['dataset_path'])
        args['skip'] = total_train_imgs // 16
        args['maxframes'] = 16
    if data_type in ['freeview', 'tpose']:
        # set skip for freeview dataset
        # from config.py
        args['skip'] = cfg.render_skip

    # get dataset class instance according to datatype
    # from freeview/movement/train .py
    dataset = _query_dataset(data_type)
    # create an instance of the dataset class.
    # initialize its constructor using args
    dataset = dataset(**args)
    # return dataset intitialized with args
    return dataset


def _worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def create_dataloader(data_type='train'):
    # eg data_type = 'freeview'
    cfg_node = cfg[data_type]
    # get batch_size, shuffle, drop_last
    # from default.yaml
    # under freeview/train/movement
    batch_size = cfg_node.batch_size
    shuffle = cfg_node.shuffle
    drop_last = cfg_node.drop_last

    # call create dataset function.
    dataset = create_dataset(data_type=data_type)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last,
                                              num_workers=cfg.num_workers,
                                              worker_init_fn=_worker_init_fn)

    return data_loader
