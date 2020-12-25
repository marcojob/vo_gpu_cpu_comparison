import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from src.datasets_helper import DatasetsHelper
from src.plot_helper import PlotHelper
from src.utils import *

class VisualOdometry:
    def __init__(self, cpu_or_gpu, detector, dataset):
        # Choose detector
        self.detector_name = detector

        # Determine if to run pipeline on GPU 
        self.on_gpu = True if cpu_or_gpu == "gpu" else False

        # Datasets helper
        self.dh = DatasetsHelper(dataset)
        if self.dh.resized_frame_size:
            self.resized_frame_size = self.dh.resized_frame_size
        else:
            self.resized_frame_size = None
        self.intrinsic_matrix = np.array(self.dh.intrinsic_matrix)

        # Init plot helper
        self.ph = PlotHelper()

        # Initialize detector
        if self.detector_name == 'FAST':
            self.detector = cv2.cuda_FastFeatureDetector.create(threshold=75, nonmaxSuppression=True)
        elif self.detector_name == 'SIFT':
            self.detector = cv2.cuda.xfeatures2d.SIFT_create(MAX_NUM_FEATURES) # FIX
        elif self.detector_name == 'SURF':
            self.detector = cv2.cuda.SURF_CUDA_create(300, _nOctaveLayers=2)
        elif self.detector_name == 'ORB':
            self.detector = cv2.cuda_ORB.create(nfeatures=MAX_NUM_FEATURES)
        elif self.detector_name == 'SHI-TOMASI':
            self.detector = cv2.cuda.createGoodFeaturesToTrackDetector(cv2.CV_8UC1, feature_params['maxCorners'], \
                                           feature_params['qualityLevel'], feature_params['minDistance'], \
                                           feature_params['blockSize'])
        elif self.detector_name == 'REGULAR_GRID':
            self.init_regular_grid_detector()
        else:
            assert 1 == 0, "Detector not available"

        # Initialize clahe filter
        self.clahe = cv2.cuda.createCLAHE(clipLimit=5.0)

        # LK
        lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,)
                  # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.lk = cv2.cuda.SparsePyrLKOpticalFlow_create(**lk_params)

        # Frames
        self.cur_c_frame = None
        self.pre_c_frame = None

        self.cur_g_frame = None
        self.pre_g_frame = None

        self.cur_rgb_c_frame = None

        # Features
        self.cur_c_fts = None  # Current CPU features
        self.pre_c_fts = None  # Previous CPU features

        self.cur_g_fts = None  # Current GPU features
        self.pre_g_fts = None  # Previous GPU features

        # Rotation, translation
        self.cur_r = np.eye(3) # Current rotation
        self.cur_t = np.zeros((3, 1)) # Current translation
        self.all_r = [self.cur_r]
        self.all_t = [self.cur_t]

        # Global pos data
        self.x_data = list()
        self.y_data = list()
        self.z_data = list()

        # Framerate tracking
        self.framerate_window = 100
        self.framerate_list = [30.0 for i in range(self.framerate_window)]
        self.framerate = 0.0
        self.framecount = 0

        # N Features tracking
        self.nfeatures_window = 20 # frames to track
        self.nfeatures_list = [0.0 for i in range(self.nfeatures_window)]

        # Masks
        self.mask_ch = None

        # Cloud
        self.cloud = None
    
    def start(self, plot=True):
        # Get first frame
        frame = next(self.dh.images)

        # Process first frame
        self.process_first_frame(frame)

        # Loop through frames
        for frame in self.dh.images:
                # Main frame processing
                self.process_frame(frame)
                self.framecount += 1

                # Plotting
                frame = self.cur_rgb_c_frame
                frame = self.draw_of(frame, self.pre_c_fts, self.cur_c_fts, self.mask_ch)
                
                self.ph.plot(frame, self.framerate, self.cloud, self.nfeatures_list, self.framecount) 

    
        # Release the capture
        cap.release()

    def draw_fts(self, frame, fts):
        size = 3
        col = (255, 0, 0)
        th = 1
        for f in fts:
            x, y = int(f[0]), int(f[1])
            frame = cv2.circle(frame, (x, y), size, col, thickness=th)
        return frame

    def draw_of(self, frame, pre_fts, cur_fts, mask):
        size = 3
        col = (0, 255, 0)
        th = 2
        for m, p, c in zip(mask, pre_fts, cur_fts):
            if m:
                end_point = (int(p[0]), int(p[1]))
                start_point = (int(c[0]), int(c[1]))

                frame = cv2.arrowedLine(frame, start_point, end_point, col, th)
        return frame

    def draw_framerate(self, frame, framerate):
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (880, 15)
        fontScale = 0.5
        col = (255, 0, 0)
        th = 2
        return cv2.putText(frame, "FPS: "+str(framerate), org, font,
                           fontScale, col, th, cv2.LINE_AA)

    def process_frame(self, frame):
        # Start timer
        process_frame_start = time.monotonic()

        # Upload resized frame to GPU
        self.gf = cv2.cuda_GpuMat()
        self.gf.upload(frame)

        # Resize frame
        if not self.resized_frame_size is None:
            self.gf = cv2.cuda.resize(self.gf, self.resized_frame_size)
            self.cur_rgb_c_frame = self.gf.download()
        else:
            self.cur_rgb_c_frame = frame

        # Convert to gray
        self.gf = cv2.cuda.cvtColor(self.gf, cv2.COLOR_BGR2GRAY)

        # Update CPU frame
        self.pre_c_frame = self.cur_c_frame
        self.cur_c_frame = self.gf.download()

        # Apply Clahe filter
        if USE_CLAHE:
            self.gf = self.clahe.apply(self.gf, None)

        # Update prev and curr img
        self.pre_g_frame = self.cur_g_frame
        self.cur_g_frame = self.gf

        # Detect new features if we don't have enough
        if len(self.pre_c_fts) < MIN_NUM_FEATURES:
            self.cur_g_fts = self.detect_new_features(self.cur_g_frame)
            self.pre_g_fts = self.detect_new_features(self.pre_g_frame)

            # Convert keypoints to CPU
            self.cur_c_fts = self.convert_fts_gpu_to_cpu(self.cur_g_fts)
            self.pre_c_fts = self.convert_fts_gpu_to_cpu(self.pre_g_fts)

            # The GPU keypoints need to be in this format for some reason
            tmp = cv2.cuda_GpuMat()
            tmp_re = self.pre_c_fts.reshape((1, -1, 2))
            tmp.upload(tmp_re)
            self.pre_g_fts = tmp

            # The GPU keypoints need to be in this format for some reason
            tmp = cv2.cuda_GpuMat()
            tmp_re = self.cur_c_fts.reshape((1, -1, 2))
            tmp.upload(tmp_re)
            self.cur_g_fts = tmp


        # Track g fts
        self.pre_g_fts = self.cur_g_fts

        # Sparse OF
        self.pre_c_fts, self.cur_c_fts, _ = self.KLT_featureTracking(self.pre_g_frame, self.cur_g_frame, self.pre_g_fts)

        # Upload to GPU also
        self.pre_g_fts.upload(self.pre_c_fts.reshape((1, -1, 2)))
        self.cur_g_fts.upload(self.cur_c_fts.reshape((1, -1, 2)))

        # Find Essential matrix
        E, mask = cv2.findEssentialMat(self.cur_c_fts, self.pre_c_fts, self.intrinsic_matrix, cv2.RANSAC, 0.99, 1.0, None)

        # Recover pose
        ret, r, t, self.mask_ch = cv2.recoverPose(E, self.cur_c_fts, self.pre_c_fts, self.intrinsic_matrix, mask)
        if ret > 10:
            # Only keep mask of features
            tmp_cur_fts = list()
            tmp_pre_fts = list()
            for i, m in enumerate(self.mask_ch):
                if m[0]:
                    tmp_cur_fts.append(self.cur_c_fts[i])
                    tmp_pre_fts.append(self.pre_c_fts[i])
            self.cur_c_fts = np.array(tmp_cur_fts)
            self.pre_c_fts = np.array(tmp_pre_fts)

            # Continue tracking of movement
            self.scale = 1.0 / np.linalg.norm(t)
            self.cur_r = r.dot(self.cur_r)  # Concatenate the rotation matrix
            self.cur_t = self.cur_t + self.scale * self.cur_r.dot(t)  # Concatenate the translation vectors

            # Triangulate points
            self.cloud = self.triangulate_points(self.cur_r, self.cur_t, r, t)

        # Update the number of features tracked
        self.nfeatures_list.append(ret)
        if len(self.nfeatures_list) > self.nfeatures_window:
            del self.nfeatures_list[0]

        # Download frame
        self.d_frame = self.gf.download()

        # End timer and compute framerate
        framerate = round(1.0 / (time.monotonic() - process_frame_start))
        self.framerate_list.append(framerate) 
        if len(self.framerate_list) > 100:
            del self.framerate_list[0]
        self.framerate = sum(self.framerate_list)/len(self.framerate_list)

    def triangulate_points(self, R, t, delta_R, delta_t):
        P0 = np.dot(self.intrinsic_matrix, np.eye(3, 4))

        P1 = np.hstack((delta_R, -delta_t))
        P1 = self.intrinsic_matrix.dot(P1)

        cloud_homo = cv2.triangulatePoints(P0, P1, self.pre_c_fts.T, self.cur_c_fts.T)
        cloud = cv2.convertPointsFromHomogeneous(cloud_homo.T).reshape(-1, 3)

        return cloud

    def KLT_featureTracking(self, prev_img, cur_img, prev_fts):
        # Feature tracking using the Kanade-Lucas-Tomasi tracker

        # Feature correspondence with backtracking
        kp2_g, status, error = self.lk.calc(prev_img, cur_img, prev_fts, None)
        kp1_g, status, error = self.lk.calc(cur_img, prev_img, kp2_g, None)

        # Get CPU keypoints
        kp2 = kp2_g.download().reshape((1, -1, 2))
        kp1 = kp1_g.download().reshape((1, -1, 2))

        # Find difference
        d = abs(prev_fts.download() - kp1).reshape(-1, 2).max(-1)
        diff = d < MIN_MATCHING_DIFF

        # Error Management
        if len(d) == 0:
            print('Error: No point correspondance.')
        # If less than 5 good points, it uses the features obtain without the backtracking check
        elif list(diff).count(True) <= 5:
            print('Warning: Few point correspondances')
            return kp1, kp2, MIN_MATCHING_DIFF

        # Create new lists with the good features
        n_kp1, n_kp2 = [], []
        for i, f in enumerate(diff):
            if f:
                n_kp1.append(kp1[0][i])
                n_kp2.append(kp2[0][i])

        # Format the features into float32 numpy arrays
        n_kp1, n_kp2 = np.array(n_kp1, dtype=np.float32), np.array(
            n_kp2, dtype=np.float32)

        # Verify if the point correspondence points are in the same pixel coordinates
        d = abs(n_kp1 - n_kp2).reshape(-1, 2).max(-1)

        # The mean of the differences is used to determine the amount of distance between the pixels
        diff_mean = np.mean(d)

        return n_kp1, n_kp2, diff_mean

    def process_first_frame(self, frame):
        # Upload resized frame to GPU
        self.gf = cv2.cuda_GpuMat()
        self.gf.upload(frame)

        # Resize frame
        if not self.resized_frame_size is None:
            self.gf = cv2.cuda.resize(self.gf, self.resized_frame_size)
            self.cur_rgb_c_frame = self.gf.download()
        else:
            self.cur_rgb_c_frame = frame

        # Convert to gray
        self.gf = cv2.cuda.cvtColor(self.gf, cv2.COLOR_BGR2GRAY)

        # Update CPU frame
        self.cur_c_frame = self.gf.download()

        # Apply Clahe filter
        if USE_CLAHE:
            self.gf = self.clahe.apply(self.gf, None)

        # Update curr img
        self.cur_g_frame = self.gf

        # Detect initial features
        self.cur_g_fts = self.detect_new_features(self.cur_g_frame)

        # Convert keypoints to CPU
        self.cur_c_fts = self.convert_fts_gpu_to_cpu(self.cur_g_fts)
        self.pre_c_fts = self.cur_c_fts

        # Reshape
        tmp = cv2.cuda_GpuMat()
        tmp_re = self.cur_c_fts.reshape((1, -1, 2))
        tmp.upload(tmp_re)
        self.cur_g_fts = tmp
        self.pre_g_fts = tmp

    def detect_new_features(self, img):
        # Detect features using selected detector
        if self.detector_name == 'FAST' or self.detector_name == 'ORB':
            g_kps = self.detector.detectAsync(img, None)
        elif self.detector_name == 'SURF':
            g_kps = self.detector.detect(img, None)
        elif self.detector_name == 'SHI-TOMASI':
            g_kps = self.detector.detect(img)
        elif self.detector_name == 'REGULAR_GRID':
            # Not very efficient, but regular grid comp. is low
            img_c = img.download()
            g_kps = self.regular_grid_detector(img_c)
        return g_kps

    def regular_grid_detector(self, img):
        return self.gpu_rg

    def init_regular_grid_detector(self):
        # Init regular grid
        self.regular_grid_max_pts = MAX_NUM_FEATURES

        features = list()
        height = float(self.dh.size[1])
        width = float(self.dh.size[0])
        k = height/width

        n_col = int(np.sqrt(self.regular_grid_max_pts/k))
        n_rows = int(n_col*k)

        h_cols = int(width/n_col)
        h_rows = int(height/n_rows)

        for c in range(n_col):
            for r in range(n_rows):
                features.append(np.array((c*h_cols, r*h_rows), dtype=np.float32))

        gpu_f = cv2.cuda_GpuMat()
        cpu_f = np.array(features, dtype=np.float32).reshape((1, -1, 2))
        gpu_f.upload(cpu_f)
        self.gpu_rg = gpu_f

    def convert_fts_gpu_to_cpu(self, g_fts):
        if self.detector_name == 'FAST' or self.detector_name == 'ORB':
            c_fts = self.detector.convert(g_fts)
            c_fts = np.array([x.pt for x in c_fts], dtype=np.float32)
        elif self.detector_name == 'SURF':
            c_fts = cv2.cuda_SURF_CUDA.downloadKeypoints(self.detector, g_fts)
            c_fts = np.array([x.pt for x in c_fts], dtype=np.float32)
        elif self.detector_name == 'SHI-TOMASI' or self.detector_name == 'REGULAR_GRID':
            # Not very efficient, but regular grid comp. is low
            c_fts = g_fts.download()

        return c_fts
