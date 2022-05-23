"""
Mask R-CNN

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet
"""

import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3" # specify which GPU(s) to be used

import sys
import json
import datetime
import numpy as np
import skimage.draw

# test ..
import scipy
from scipy import misc
import matplotlib.pyplot as plt
import imgaug

import cv2
import glob

from keras.callbacks import TensorBoard

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn_crack.config import Config
from mrcnn_crack import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Select data and label types 
DATA_TYPE = "uav" # "public", "uav", "test"
LABEL_TYPE= "instance_mask" # "whole_mask", "instance_mask", "test"



############################################################
#  Configurations
############################################################

class Configuration(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "crack"
    
    # Can override..? yes. make sure for batch_size!
    GPU_COUNT = 2

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + target)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 200
    LEARNING_RATE = 0.001

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 256 
    IMAGE_MAX_DIM = 256

    # Weight decay regularization
    WEIGHT_DECAY = 0.001
    
    # For soft NMS
    SCORE_THRESHOLD = 0.1
    SOFT_NMS_SIGMA = 0.5
    
    
############################################################
#  Dataset
############################################################

class Datasets(utils.Dataset):
    def load_data(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("crack", 1, "crack")
        
        # datatype: Either "public" (for public dataset) or "uav" (UAV or Dr.Won dataset)
        self.datatype = DATA_TYPE

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        
        if self.datatype == "public":
            if subset == "train":
                subset = "traincrop/processed"
                dataset_dir = os.path.join(dataset_dir, subset)
            elif subset == "val":
                subset = "valcrop/processed"
                dataset_dir = os.path.join(dataset_dir, subset)
            else:
                dataset_dir = os.path.join(dataset_dir)
            
            
        elif self.datatype == "uav":
            if subset == "train":
                subset = "train"
                dataset_dir = os.path.join(dataset_dir, subset)
            elif subset == "val":
                subset = "val"
                dataset_dir = os.path.join(dataset_dir, subset)
            else:
                dataset_dir = os.path.join(dataset_dir)
                
        elif self.datatype == "test":
            dataset_dir = os.path.join(dataset_dir)
            
        image_ids = next(os.walk(dataset_dir))[2]
        
        for image_id in image_ids:
            if (self.datatype == "public" and image_id.endswith(".jpg")) or (self.datatype == "uav" and image_id.endswith(".jpeg")) or (self.datatype == "test" and (image_id.endswith(".bmp") or image_id.lower().endswith(".jpg"))):
                self.add_image(
                    "crack",
                    image_id=image_id,
                    path=os.path.join(dataset_dir, image_id)
                    )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        
        
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(info['path']))
        filename = os.path.basename(info['path'])
        
        # Read mask files from .png image
        mask = []
        
        # label_mode: Either "whole_mask" (for one ground truth mask per image) or "instance_mask" (for multiple ground truth masks per image)
        self.label_mode = LABEL_TYPE
        
        if self.label_mode == "whole_mask":
            for f in next(os.walk(mask_dir))[2]:
                #if f.endswith(".png"):
                if (self.datatype == "public" and f == (filename[:-4]+".png")) or (self.datatype == "uav" and f == (filename[:-5]+".png")): 
                    m = cv2.imread(os.path.join(mask_dir, f), 0).astype(np.bool)
                    #m = skimage.io.imread((self.dataset_dir+image_id), as_gray=True, plugin='matplotlib')
                    mask.append(m)
            mask = np.stack(mask, axis=-1)
        
        
        elif self.label_mode == "instance_mask":
            for f in next(os.walk(mask_dir))[2]:
                #if f == (filename[:-5]+".npy"):
                if f == (os.path.splitext(filename)[0]+".npy"):
                    #m = cv2.imread(os.path.join(mask_dir, f), 0).astype(np.bool)
                    mask = np.load(os.path.join(mask_dir, f)).astype(np.bool) #.tolist()
                    #m = skimage.io.imread((self.dataset_dir+image_id), as_gray=True, plugin='matplotlib')
                    #mask.append(m)
            #mask = np.stack(m, axis=-1)
            
        elif self.label_mode == "test":
            for f in next(os.walk(mask_dir))[2]:
                #if f == (os.path.splitext(filename)[0]+".bmp"):
                _m = cv2.imread(os.path.join(mask_dir, f), 0).astype(np.bool)
                m = np.zeros(_m.shape).astype(np.bool)
                mask.append(m)
            mask = np.stack(mask, axis=-1)
        
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
        
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "crack":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = Datasets()
    dataset_train.load_data(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = Datasets()
    dataset_val.load_data(args.dataset, "val")
    dataset_val.prepare()

    # Data augmentation
    augmentation = imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.SomeOf(2, [imgaug.augmenters.Fliplr(0.5),
                                                                         imgaug.augmenters.Flipud(0.5),
                                                                         imgaug.augmenters.Affine(rotate=90),
                                                                         imgaug.augmenters.Superpixels(p_replace=0.5, n_segments=64),
                                                                         imgaug.augmenters.GaussianBlur(sigma=(0.0, 3.0))]))        
    
    # Training schedule
    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=200, 
            layers="heads",
            augmentation=augmentation)
    
    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 100,
            epochs=400, 
            layers="all",
            augmentation=augmentation)
    


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect bolts.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        default='./../../dataset/uav_data/data/',
                        metavar="/path/to/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=False,
                        default='coco',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = Configuration()
    else:
        class InferenceConfig(Configuration):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
