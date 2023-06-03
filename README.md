# HumanNeRF: Free-viewpoint Rendering of Moving People from Monocular Video (CVPR 2022)

## Change Log
 - Updated Files with comments on what the functions do (OnGoing).
## To_Do
- [ ] Find out what the motion_weight_prior(gaussian bone volume work).
- [ ] Understand the tpose.py dataset class and the tpose rendering process from run.py
- [ ] Find out what Rh and Th exactly are ( its given they are human orientations in one of the issues)._
- [ ] Why do they pass Rh again to apply_global_tfm_to_camera after getting the updated E matrix according to the frame_indx.
- [ ] What does apply_global_tfm_to_camera exactly do ?

## File Structures

1. mesh_infos.pkl
    - 'frame_000000'
        - 'Rh'           : np.array(3,)
        - 'Th'           : np.array(3,)
        - 'poses'        : np.array(72,)
        - 'joints'       : np.array(24,3)
        - 'tpose_joints' : np.array(24,3) 
    - 'frame_000001'
        - ...

2. camera.pkl
    - 'frame_000000'
        - 'intrinsics'  : np.array(3,3)
        - 'extrinsics'  : np.array(4,4)
        - 'distortions' : np.array(3,3)
    - 'frame_000001'
        - ....
    - *Problem : for all frame_xxxxxx everthing is same ?*

3. canonical_joints.pkl
    - 'joints' : np.array(24,3)