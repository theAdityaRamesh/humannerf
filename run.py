import os

import torch
import numpy as np
from tqdm import tqdm

from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image

from configs import cfg, args

# exclude these from passing on to gpu
# in cpu_data_to_gpu_function
EXCLUDE_KEYS_TO_GPU = ['frame_name',
                       'img_width', 'img_height', 'ray_mask']


def load_network():
    # create an instance of the network class
    model = create_network()
    # load the checkpoint latest.tar
    ckpt_path = os.path.join(cfg.logdir, f'{cfg.load_net}.tar')
    # load model parameters of latest.tar into ckpt variable
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    # assign loaded params to the model variable
    model.load_state_dict(ckpt['network'], strict=False)
    # print path to checkpoint
    print('load network from ', ckpt_path)
    # return the model
    return model.cuda().deploy_mlps_to_secondary_gpus()


def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    # create empty np array with zero as entry of size h*w
    alpha_map = np.zeros((height * width), dtype='float32')
    # put value of alpha obtained from 
    # model forward pass in places give by
    # ray_mask
    alpha_map[ray_mask] = alpha_vals
    # return np array of dim (h,w)
    return alpha_map.reshape((height, width))


def unpack_to_image(width, height, ray_mask, bgcolor,
                    rgb, alpha, truth=None):
    
    # numpy.full(shape, fill_value, dtype=None, order='C', *, like=None)
    # return np array of dim (h*w,3) and entries = bgcolor
    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    truth_image = np.full((height * width, 3), bgcolor, dtype='float32')

    # assign color from rgb variable 
    # to rgb_image var to places given by
    # the ray_mask
    rgb_image[ray_mask] = rgb
    # clip values of image b/w 0->255
    # convert img data-type to uint8
    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))

    # unuser code ----------------
    if truth is not None:
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))
    # ----------------------------

    # get single channel alpha image of dim (h,w)
    alpha_map = unpack_alpha_map(alpha, ray_mask, width, height)
    # return 3 channel version of alpha image
    # converted to uint8 format b/w 0->255
    alpha_image  = to_8b3ch_image(alpha_map)

    return rgb_image, alpha_image, truth_image


def _freeview(
        data_type='freeview',
        folder_name=None):
    
    # _freeview(
    #     data_type='freeview',
    #     folder_name=f"freeview_{cfg.freeview.frame_idx}" \
    #         if not cfg.render_folder_name else cfg.render_folder_name)

    cfg.perturb = 0.

    # load the latest saved model
    # from checkpoint
    model = load_network()
    # create the test dataloader
    # with data-type ='freeview'
    test_loader = create_dataloader(data_type)
    # create an instance of the image writer class
    # pass the output directory to constructor
    writer = ImageWriter(
                output_dir=os.path.join(cfg.logdir, cfg.load_net),
                exp_name=folder_name)

    # set model to eval
    model.eval()

    # iterate over the batch items in the test
    # data loader
    for batch in tqdm(test_loader):
        for k, v in batch.items():
            ## ??????????
            # why do we do this find out ?
            batch[k] = v[0]

        # lift data of batch from cpu to gpu
        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU)

        # stop backprop 
        # compute model forward pass
        # for the data variable 
        # containing batch items
        with torch.no_grad():
            net_output = model(**data, 
                               iter_val=cfg.eval_iter)
        # get rgb colors and alpha
        # for the particular 3d point
        # from the model forward pass
        rgb = net_output['rgb']
        alpha = net_output['alpha']
        # get the image height, width info 
        # from the batch 
        width = batch['img_width']
        height = batch['img_height']
        # WHAT IS RAY MASK ???
        # ??? find out
        ray_mask = batch['ray_mask']
        # if target rgb present get it
        # otherwise return None
        target_rgbs = batch.get('target_rgbs', None)

        # get unit8 formatted images
        # of dim (h,w) for rgb and alpha
        rgb_img, alpha_img, _ = unpack_to_image(
            width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
            rgb.data.cpu().numpy(),
            alpha.data.cpu().numpy())

        # append rgb image to img
        imgs = [rgb_img]
        # if show_truth : True in default.yaml
        # append ground truth image
        if cfg.show_truth and target_rgbs is not None:
            target_rgbs = to_8b_image(target_rgbs.numpy())
            imgs.append(target_rgbs)
        # if show_apha : True in default.yaml
        # append alpha image 
        if cfg.show_alpha:
            imgs.append(alpha_img)

        # concatenate all three images
        # rgb_image(NVS) | ground_truth | alpha_map
        img_out = np.concatenate(imgs, axis=1)
        # add img_out to image writer class
        # save the img_out variable as a png image
        writer.append(img_out)

    # does nothing
    writer.finalize()


def run_freeview():
    # call the _freeview function
    # if cfg.render_folder_name is already assigned
    # in config.py then user the name cfg.freeview.frame_idx
    # otherwise use the assigned name in config.py
    _freeview(
        data_type='freeview',
        folder_name=f"freeview_{cfg.freeview.frame_idx}" \
            if not cfg.render_folder_name else cfg.render_folder_name)


def run_tpose():
    cfg.ignore_non_rigid_motions = True
    _freeview(
        data_type='tpose',
        folder_name='tpose' \
            if not cfg.render_folder_name else cfg.render_folder_name)


def run_movement(render_folder_name='movement'):
    cfg.perturb = 0.
    # create instance of network 
    model = load_network()
    # create dataloader of type movement
    test_loader = create_dataloader('movement')
    writer = ImageWriter(
        output_dir=os.path.join(cfg.logdir, cfg.load_net),
        exp_name=render_folder_name)

    model.eval()
    for idx, batch in enumerate(tqdm(test_loader)):
        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])

        with torch.no_grad():
            net_output = model(**data, iter_val=cfg.eval_iter)

        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']

        rgb_img, alpha_img, truth_img = \
            unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor)/255.,
                rgb.data.cpu().numpy(),
                alpha.data.cpu().numpy(),
                batch['target_rgbs'])

        imgs = [rgb_img]
        if cfg.show_truth:
            imgs.append(truth_img)
        if cfg.show_alpha:
            imgs.append(alpha_img)
            
        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out, img_name=f"{idx:06d}")
    
    writer.finalize()

        
if __name__ == '__main__':
    # call the function
    # run_freeview/tpose/movement
    globals()[f'run_{args.type}']()
