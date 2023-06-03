# HumanNeRF: Free-viewpoint Rendering of Moving People from Monocular Video (CVPR 2022)

## Change Log
 - Updated Files with comments on what the functions do (OnGoing).
## To_Do
- [ ] Find out what the motion_weight_prior(gaussian bone volume work).
- [ ] Understand the tpose.py dataset class and the tpose rendering process from run.py
- [ ] Find out what Rh and Th exactly are ( its given they are human orientations in one of the issues)._
- [ ] Why do they pass Rh again to apply_global_tfm_to_camera after getting the updated E matrix according to the frame_indx.
- [ ] What does apply_global_tfm_to_camera exactly do ?