import os
import shutil

from termcolor import colored
from PIL import Image
import numpy as np


def load_image(path, to_rgb=True):
    img = Image.open(path)
    return img.convert('RGB') if to_rgb else img


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def to_8b_image(image):
    # numpy.clip(a, a_min, a_max, out=None, **kwargs)
    return (255.* np.clip(image, 0., 1.)).astype(np.uint8)


def to_3ch_image(image):
    # if image.shape =(h,w)
    if len(image.shape) == 2:
        # stack the image thrice
        return np.stack([image, image, image], axis=-1)
        # if image shape is (h,w,1)
    elif len(image.shape) == 3:
        # check if 3rd dim is = 1
        assert image.shape[2] == 1
        # concatenate image
        return np.concatenate([image, image, image], axis=-1)
    else:
        print(f"to_3ch_image: Unsupported Shapes: {len(image.shape)}")
        return image


def to_8b3ch_image(image):
    # convert input to unit8 format b/w 0->255
    # return 3channel image of that image
    return to_3ch_image(to_8b_image(image))


def tile_images(images, imgs_per_row=4):
    rows = []
    row = []
    imgs_per_row = min(len(images), imgs_per_row)
    for i in range(len(images)):
        row.append(images[i])
        if len(row) == imgs_per_row:
            rows.append(np.concatenate(row, axis=1))
            row = []
    if len(rows) > 2 and len(rows[-1]) != len(rows[-2]):
        rows.pop()
    imgout = np.concatenate(rows, axis=0)
    return imgout

     
class ImageWriter():
    # take as input to constructor
    # output directory
    # experiment name
    def __init__(self, output_dir, exp_name):
        # create image directory path
        # from output directory and experiment name.
        self.image_dir = os.path.join(output_dir, exp_name)

        print("The rendering is saved in " + \
              colored(self.image_dir, 'cyan'))
        
        # remove image dir if it exists
        if os.path.exists(self.image_dir):
            shutil.rmtree(self.image_dir)
        
        # create a directory at path image_dir
        os.makedirs(self.image_dir, exist_ok=True)
        self.frame_idx = -1

    def append(self, image, img_name=None):
        # increase frame indx by one 
        # on appending image to Image Writer
        self.frame_idx += 1
        if img_name is None:
            # if no image name is provided
            # name the images in order of 
            # which they were appended
            img_name = f"{self.frame_idx:06d}"
        # save the image as png
        save_image(image, f'{self.image_dir}/{img_name}.png')
        # return frame indx and image name.
        return self.frame_idx, img_name

    # unused code
    def finalize(self):
        pass
