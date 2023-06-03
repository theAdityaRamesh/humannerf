import os
import pickle

import numpy as np
import cv2
import torch
import torch.utils.data

from core.utils.image_util import load_image
from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from core.utils.camera_util import \
    rotate_camera_by_frame_idx, \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox
from core.utils.file_util import list_files, split_path

from configs import cfg


class Dataset(torch.utils.data.Dataset):
    # parameters to rotate the camera.
    ROT_CAM_PARAMS = {
        'zju_mocap': {'rotate_axis': 'z', 'inv_angle': True},
        'wild': {'rotate_axis': 'y', 'inv_angle': False}
    }

    def __init__(
            self, 
            dataset_path,
            keyfilter=None,
            maxframes=-1,
            skip=1,
            bgcolor=None,
            src_type="zju_mocap",
            **_):
        
        print('[Dataset Path]', dataset_path) 
        # datasetpath from dataset_args.py
        self.dataset_path = dataset_path
        # image directory in dataset_path/images
        self.image_dir = os.path.join(dataset_path, 'images')
        # load cannonical joints and bbox 
        # obtained from cononical skeleton
        self.canonical_joints, self.canonical_bbox = \
            self.load_canonical_joints()
        
        # ???????
        # go to this function and write comments
        if 'motion_weights_priors' in keyfilter:
            self.motion_weights_priors = \
                approx_gaussian_bone_volumes(
                    self.canonical_joints, 
                    self.canonical_bbox['min_xyz'],
                    self.canonical_bbox['max_xyz'],
                    grid_size=cfg.mweight_volume.volume_size).astype('float32')

        # load cameras from cameras.pkl
        cameras = self.load_train_cameras()
        # load mesh_infos dict from mesh_info.pkl
        mesh_infos = self.load_train_mesh_infos()
        # get train frames from dataset/zju_mocap/images
        framelist = self.load_train_frames() 

        # reduce number of frames if skip > 1
        # skip is from core.data.create_dataset.create_dataset
        # args['skip'] = cfg.render_skip 
        # default skip = 1
        self.framelist = framelist[::skip]
        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]  

        # get freeview_frame_indx from
        # python run.py \
        # --type freeview \
        # --cfg configs/human_nerf/zju_mocap/387/adventure.yaml \
        # freeview.frame_idx 128
        self.train_frame_idx = cfg.freeview.frame_idx
        print(f' -- Frame Idx: {self.train_frame_idx}')

        # set in core.configs.config.py file        
        self.total_frames = cfg.render_frames
        print(f' -- Total Rendered Frames: {self.total_frames}')

        # get the image file name at the self.train_frame_idx location
        # in the framelist from dataset/zju_mocap/images
        self.train_frame_name = framelist[self.train_frame_idx]
        # get the camera for the seld.train_frame_idx location
        # from the cameras.pkl file
        self.train_camera = cameras[framelist[self.train_frame_idx]]
        # get the train mesh_info for the self.train_frame_idx location
        # from the mesh_info.pkl file
        self.train_mesh_info = mesh_infos[framelist[self.train_frame_idx]]
        # keyfilter, src_type from dataset_args.py
        # bgcolor from adventure.yaml
        self.bgcolor = bgcolor if bgcolor is not None else [255., 255., 255.]
        self.keyfilter = keyfilter
        self.src_type = src_type
        ##
        # end of constructor

    def load_canonical_joints(self):
        # create the path to canonical joints pkl file
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        # open the pkl file
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        # get the canonical joints from the .pkl file
        canonical_joints = cl_joint_data['joints'].astype('float32')
        # create a bbox from canonical skeleton
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)
        # return bbox and joints
        return canonical_joints, canonical_bbox

    def load_train_cameras(self):
        # set camera object to empty
        cameras = None
        # set camera object to contents of cameras.pkl file
        with open(os.path.join(self.dataset_path, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        # return cameras
        return cameras

    @staticmethod
    # convert skeleton to bbox by taking the min and max 
    # of the skeleton coordinates and +/- a offset to ensure extra
    # space
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_mesh_infos(self):
        mesh_infos = None
        with open(os.path.join(self.dataset_path, 'mesh_infos.pkl'), 'rb') as f:   
            mesh_infos = pickle.load(f)
        # get frame_name from mesh_info.keys()
        for frame_name in mesh_infos.keys():
            # set bbox to bbox obtained from the joints for each frame
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            # for each frame store bbox as a dict with
            # key 'bbox' and value bbox
            # in the frame_name th index of the mesh_infos list
            mesh_infos[frame_name]['bbox'] = bbox
        # return mesh_infos for all frames as list
        return mesh_infos

    def load_train_frames(self):
        # go to dataset/zju_mocap/{sub}/images
        # get list of all files
        # cocatenate and return paths in img_paths var
        img_paths = list_files(os.path.join(self.dataset_path, 'images'),
                               exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]
    
    def query_dst_skeleton(self):
        #create a dict of these entries and return it
        return {
            'poses': self.train_mesh_info['poses'].astype('float32'),
            'dst_tpose_joints': \
                self.train_mesh_info['tpose_joints'].astype('float32'),
            'bbox': self.train_mesh_info['bbox'].copy(),
            'Rh': self.train_mesh_info['Rh'].astype('float32'),
            'Th': self.train_mesh_info['Th'].astype('float32')
        }

    def get_freeview_camera(self, frame_idx, total_frames, trans=None):
        # for freeview rendering rotate the camera around the subject
        # w.r.t to number of frames, ie: camera position is a function
        # of number of rendered frames.
        # get extrinsic camera location using 
        # rotate_camera_by_frame_idx 
        # function.
        E = rotate_camera_by_frame_idx(
                extrinsics=self.train_camera['extrinsics'], 
                frame_idx=frame_idx,
                period=total_frames,
                trans=trans,
                **self.ROT_CAM_PARAMS[self.src_type])
        K = self.train_camera['intrinsics'].copy()
        K[:2] *= cfg.resize_img_scale
        # return updated K and E camera pose matrices
        # corresponding to the frame index
        return K, E

    def load_image(self, frame_name, bg_color):
        '''
            create a image
            without background
            return resized image
        '''
        # imagepath = dataset/zju_mocap/{sub}/images/{frame_name.png}
        imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
        # load image present at image path
        orig_img = np.array(load_image(imagepath))
        # get maskpath
        maskpath = os.path.join(self.dataset_path, 
                                'masks', 
                                '{}.png'.format(frame_name))
        # load the mask
        alpha_mask = np.array(load_image(maskpath))
        
        # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        # correct for lens distortions
        if 'distortions' in self.train_camera:
            K = self.train_camera['intrinsics']
            D = self.train_camera['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)
        # convert range alpha_mask to (0,1)
        alpha_mask = alpha_mask / 255.
        # mask out the background pixels by using the alpha mask
        # colour the non-foregroung pixels to bg_color
        img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        # resize image and alpha mask accr to factor in adventure.yaml
        # WHY ??
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                             fx=cfg.resize_img_scale,
                             fy=cfg.resize_img_scale,
                             interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
        # return the image and alpha_mask
        return img, alpha_mask

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        # idx represents one of the {1,cfg.render_frames}
        # number of frames, the freeview frame is already
        # fixed from the cfg.freeview.frame_idx argument
        # given through cmdline.

        # train_frame_name = framelist[self.frame_idx]
        frame_name = self.train_frame_name
        # freeview Dataset class returns a dict results
        # set key value pair 'frame_name' : frame_name
        results = {
            'frame_name': frame_name
        }
        # bgcolor from adventure.yaml
        bgcolor = np.array(self.bgcolor, dtype='float32')
        # load the corresponding image from the frame_name
        # with background == bgcolor
        img, _ = self.load_image(frame_name, bgcolor)
        # clip image b/w (0,1)
        img = img / 255.
        # get height and width of image
        H, W = img.shape[0:2]
        # get the dataset skeleton info
        dst_skel_info = self.query_dst_skeleton()
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses']
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']
        dst_Rh = dst_skel_info['Rh']
        dst_Th = dst_skel_info['Th']
        # get the K,E matrices according 
        # to the frame index.
        K, E = self.get_freeview_camera(
                        frame_idx=idx,
                        total_frames=self.total_frames,
                        trans=dst_Th)
        # use the E matrix to apply camera transformation
        # and get the updated camera pose
        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_Rh,
                Th=dst_Th)
        R = E[:3, :3]
        T = E[:3, 3]
        
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')
    
        batch_rays = np.stack([rays_o, rays_d], axis=0) 

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor})

        if 'target_rgbs' in self.keyfilter:
            results['target_rgbs'] = img

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints)
            cnl_gtfms = get_canonical_global_tfms(self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms
            })                                    

        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = \
                self.motion_weights_priors.copy()

        if 'cnl_bbox' in self.keyfilter:
            min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
            max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
            })


        return results
