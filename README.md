# HumanNeRF: Free-viewpoint Rendering of Moving People from Monocular Video (CVPR 2022)

## Change Log
 - Updated Files with comments on what the functions do (OnGoing).
## Evaluation Scheme for QA metrics with models for different input dataset size
- Generate same test frames for all the models at every 10k iterations
- Use 7 Camera views sampled uniformly spatially (0,3,6,9 ... 21)
- Run QA metrics for each of the rendered frames.

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
4. annots.npy
    - 'cams'
        - 'K' : list(np.array(3,3)), len(list()) = 23 
        - 'R' : list(np.array(3,3)), len(list()) = 23
        - 'D' : list(np.array(5, )), len(list()) = 23
        - 'T' : list(np.array(3, )), len(list()) = 23
    - 'ims' {num_frames} items
        - 0
            - 'ims'
                - 'Camera_B1/000000.jpg'
                - 'Camera_B2/000000.jpg'
                - ...
                - 'Camera_B23/000000.jpg'
            - 'kpts2d' : (23 items)
                - np.array(25,1)
                - ...
                - np.array(25,1)
        - 1 
        - ...
        - {num_frames}
            
