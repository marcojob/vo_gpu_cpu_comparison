from pathlib import Path
import glob
import re
import cv2
import numpy as np

from datasets.configs import configs

class DatasetsHelper():
    dataset_dir = Path(__file__).parents[1] / "datasets"
    i = 0 # Number of images counter

    def __init__(self, name="KITTI"):
        config = configs.get(name, None)
        assert config != None, "Could not find the specified dataset config"

        self.intrinsic_matrix = config.get("intrinsic_matrix", None)
        assert self.intrinsic_matrix != None, "Intrinsic matrix not in config"

        self.images_dir = config.get("images_dir", None)
        self.video_file = config.get("video_file", None)

        assert self.images_dir != None or self.video_file != None, "Images dir or video file not in config"
        
        self.frame_size = config.get("frame_size", None)
        assert self.frame_size != None, "Frame size not in config"

        # Resized frame size can be None, which means not resized
        self.resized_frame_size = config.get("resized_frame_size", None)
        
        # Image pattern can be None, which mean images are just taken in alphanumerical order
        self.images_pattern = config.get("images_pattern", None)
    
    @property
    def images(self):
        """
        Iterates over all images in the dir and creates a generator as property
        """
        if self.images_dir:
            fl = glob.glob(str(self.dataset_dir / self.images_dir) + "/*")
            for f in sorted(fl):
                if self.images_pattern:
                    if re.search(self.images_pattern, f):
                        self.i += 1
                        yield cv2.imread(f)
                else:
                    self.i += 1
                    yield cv2.imread(f)
        elif self.video_file:
            cap = cv2.VideoCapture(str(self.dataset_dir / self.video_file))
            ret = 1
            while ret:
                ret, img = cap.read()
                if ret == True:
                    self.i += 1
                    yield np.copy(img)


    @property
    def size(self):
        """
        Convenient shape propery handling resizing
        """
        if self.resized_frame_size:
            return self.resized_frame_size
        else:
            return self.frame_size

