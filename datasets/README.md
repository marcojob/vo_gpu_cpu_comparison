# Datasets instructions
In order to run the script, a datasets must be provided. So far, the KITTI, the Malaga and arbitrary video datasets are supported. In the following the structure for the different kind of datasets is explained. If you want to use a different folder structure, adapt the corresponding global variable in the `datasets_helper.py` file (instructions below).

## KITTI
Download the KITTI dataset from the original source or [here](http://rpg.ifi.uzh.ch/docs/teaching/2016/kitti00.zip) and extract the content into the `datasets` folder. The resulting folder structure to the images should then be `datasets/kitti00/kitti/00/image_0`.

## Malaga
Download the Malaga dataset from the original source or [here](http://rpg.ifi.uzh.ch/docs/teaching/2016/malaga-urban-dataset-extract-07.zip) and extract the content into the `datasets` folder. The resulting folder structure to the image should then be `datasets/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_1024x768_Images`.

## Parking garage
Download the Parking garage dataset from [here](http://rpg.ifi.uzh.ch/docs/teaching/2016/parking.zip) and extract the content into the `datasets` folder. The resulting folder structure to the image should then be `datasets/parking/images`.

## Arbitrary video datasets
Place any video file into the datasets folder and add a new entry in the `datasets_helper.py` file (instructions below).

# `datasets/configs.py`: Adding / modifying entries
To add new datasets follow the already given configs. Datasets from images and from video files are supported. 
