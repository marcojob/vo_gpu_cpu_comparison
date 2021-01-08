configs = {
    "KITTI": {
        "images_dir": "kitti00/kitti/00/image_0",
        "images_pattern": "[0-9]*.png",
        "resized_frame_size": None,
        "frame_size": (1241, 376),
        "intrinsic_matrix": [[7.188560000000e+02, 0.0, 6.071928000000e+02],
                             [0.0, 7.188560000000e+02, 1.852157000000e+02],
                             [0.0, 0.0, 1.0]]
    },
    "Malaga": {
        "images_dir": "malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_1024x768_Images",
        "images_pattern": "img_CAMERA1_([0-9]*\.[0-9]+|[0-9]+)_left.jpg",
        "resized_frame_size": (512, 384),
        "frame_size": (1024, 768),
        "intrinsic_matrix": [[795.11588, 0.0, 517.12973],
                             [0.0, 795.11588, 395.59665],
                             [0.0, 0.0, 1.0]]
    },
    "Autobahn": {
        "video_file": "autobahn.MOV",
        "resized_frame_size": (480, 270),
        "frame_size": (1920, 1080),
        "intrinsic_matrix": [[1.79817676e+03, 0.00000000e+00, 9.03319760e+02],
                             [0.00000000e+00, 1.79429840e+03, 5.45307608e+02],
                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    },
    "Parking": {
        "images_dir": "parking/images",
        "images_pattern": "img_([0-9]*).png",
        "frame_size": (640, 480),
        "intrinsic_matrix": [[331.37, 0.0, 320.0],
                             [0.0, 369.568, 240.0],
                             [0.0, 0.0, 1.0]]
    }
}


