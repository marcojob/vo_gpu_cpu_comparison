import cv2
import time
import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

from src.datasets_helper import DatasetsHelper
from src.plot_helper import PlotHelper
from src.utils import *
from src.profiler import Profiler

logger = logging.getLogger("vo")

class VisualOdometry:
    def __init__(self, args):
        # Choose detector
        self.detector_name = args.detector

        # Determine if to run pipeline on GPU
        self.on_gpu = args.gpu

        # Datasets helper
        self.dh = DatasetsHelper(args.dataset)
        if self.dh.resized_frame_size:
            self.resized_frame_size = self.dh.resized_frame_size
        else:
            self.resized_frame_size = None

        # Get the intrinsic matrix
        self.intrinsic_matrix = np.array(self.dh.intrinsic_matrix)

        # Initialize detector
        if self.detector_name == 'REGULAR_GRID':
            self.init_regular_grid_detector()
        else:
            assert 1 == 0, "Detector not available"

        # LK
        lk_params = dict( winSize  = (21, 21),
                  maxLevel = 6,)
                  # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        if self.on_gpu:
            self.lk = cv2.cuda.SparsePyrLKOpticalFlow_create(**lk_params)
        else:
            self.lk = cv2.SparsePyrLKOpticalFlow_create(**lk_params)
        
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
        self.all_t = [[0.0],[0.0],[0.0]] # XYZ convention

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
        self.nfeatures_window = 50 # frames to track
        self.nfeatures_list = [0.0 for i in range(self.nfeatures_window)]
        self.nfts = list()

        # Masks
        self.mask_ch = None

        # Cloud
        self.point_cloud = None
        self.fts_color = None

        # Initialize a profiler
        Profiler.set_property_dataset(args.dataset)
        Profiler.set_property_detector(args.detector)
        Profiler.set_property_gpu(args.gpu)

        ## Start the pipeline
        self.start(plot= not args.headless)

    def start(self, plot=True):

        # Loop through frames
        for frame in self.dh.images:
            self.cur_rgb_frame = np.copy(frame)

            # Preprocess new frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Assign new frame
            self.cur_c_frame = np.copy(frame)

            # If prev frame is not assigned, gather another frame
            if self.pre_c_frame is None:
                self.finalize_step()
                continue

            # If we don't have a point cloud, compute it
            if self.point_cloud is None or self.point_cloud.shape[1] < MIN_NUM_FEATURES:
                if self.point_cloud is not None:
                    logger.warning("WARNING: Augmented pointcloud due to low number of points %d", self.point_cloud.shape[1])
                    R, t = self.bootstrap(augment=True)
                else:
                    R, t = self.bootstrap(augment=False)

                self.track_trajectory(R, t)

                # Plot
                f = self.draw_of(self.cur_rgb_frame, self.pre_c_fts, self.cur_c_fts)
                cv2.imshow("frame",f )
                cv2.waitKey()

                self.finalize_step()
                continue

            # Standard VO starts with feature tracking
            ret_klt, self.pre_c_fts, self.cur_c_fts, self.point_cloud = self.KLT_tracking(self.pre_c_frame,
                self.cur_c_frame, self.pre_c_fts, self.point_cloud)
            if ret_klt:
                logger.error("Could not do KLT tracking")

            # Apply EPnP
            R, t = self.do_pnp(True)

            # Track pose
            self.track_trajectory(R, t)
            print("Cur t", self.cur_t)

            # Plot
            f = self.draw_of(self.cur_rgb_frame, self.pre_c_fts, self.cur_c_fts)
            cv2.imshow("frame",f )
            cv2.waitKey()

            # Track frames and features
            self.finalize_step()

    def bootstrap(self, augment=False):
        # Get initial features in first frame
        new_fts = self.detect_new_features(self.pre_c_frame)
        if augment:
            self.pre_c_fts = np.concatenate((self.pre_c_fts, new_fts), axis=1)
        else:
            self.pre_c_fts = new_fts

        # Apply KLT tracking algorithm
        ret_klt, self.pre_c_fts, self.cur_c_fts, _ = self.KLT_tracking(self.pre_c_frame, self.cur_c_frame, self.pre_c_fts)
        if ret_klt:
            logger.error("Could not bootstrap")
            return ret_klt

        # Recover Essential matrix
        E, mask = cv2.findEssentialMat(self.cur_c_fts, self.pre_c_fts, self.intrinsic_matrix, cv2.RANSAC, 0.95)

        # Recover pose from essential matrix
        ret, r, t, mask_ch = cv2.recoverPose(E, self.cur_c_fts, self.pre_c_fts, self.intrinsic_matrix, mask)

        # Kick out points that did not pass cheirality check
        # self.cur_c_fts = self.cur_c_fts[:, mask_ch[:, 0], :]
        # self.pre_c_fts = self.pre_c_fts[:, mask_ch[:, 0], :]

        # Get pointcloud, if enough corresp. used
        if ret > 10:
            # TODO: Does one have to use mask_ch here?
            self.point_cloud = self.triangulate_points(r, t)
        else:
            logger.warning("Recover pose failed due to low correspondances: %d", ret)

        return r, t

    def track_trajectory(self, R, t):
        self.scale = 1.0 / np.linalg.norm(t)
        self.cur_r = self.cur_r.dot(R)
        self.cur_t = self.cur_t + self.scale * self.cur_r.dot(t)  # Concatenate the translation vectors
        self.all_t[0].append(self.cur_t[0])
        self.all_t[1].append(self.cur_t[2])
        self.all_t[2].append(self.cur_t[1])
        self.all_r.append(self.cur_r)


    def finalize_step(self):
        self.pre_c_frame = self.cur_c_frame

    def triangulate_points(self, delta_R, delta_t):
        # Perspective 0 has no rotation and no translation
        P0 = np.dot(self.intrinsic_matrix, np.eye(3, 4))

        # Perspective 1 has calculated rotation and translation
        P1 = np.hstack((delta_R, -delta_t))
        P1 = self.intrinsic_matrix.dot(P1)

        # Get cloud
        cloud_homogenous = cv2.triangulatePoints(P0, P1, self.pre_c_fts.T, self.cur_c_fts.T)
        cloud = cv2.convertPointsFromHomogeneous(cloud_homogenous.T).reshape(1, -1, 3)

        return cloud

    def KLT_tracking(self, prev_img, cur_img, prev_fts, point_cloud=None):
        kp2, status, error = self.lk.calc(prev_img, cur_img, prev_fts, None)
        kp1, status, error = self.lk.calc(cur_img, prev_img, kp2, None)

        # Format keypoints
        kp2 = kp2.reshape((1, -1, 2))
        kp1 = kp1.reshape((1, -1, 2))

        # Find difference
        d = abs(prev_fts - kp1).reshape(-1, 2).max(-1)
        diff = d < MIN_MATCHING_DIFF

        # Error Management
        if np.sum(diff) < 4:
            helper_vector = np.zeros((1, 1, 2))
            return 1, helper_vector, helper_vector, point_cloud

        # Create new array with features that fullfil check
        n_kp1 = kp1[:, diff, :]
        n_kp2 = kp2[:, diff, :]

        # Also only take pointcloud points that full this
        if point_cloud is not None:
            point_cloud = point_cloud[:, diff, :]

        return 0, n_kp1, n_kp2, point_cloud

    def detect_new_features(self, img):
        # Detect features using selected detector
        c_kps = self.regular_grid_detector()
        return c_kps

    def regular_grid_detector(self):
        if self.on_gpu:
            return self.gpu_rg
        else:
            return self.cpu_rg

    def init_regular_grid_detector(self):
        # TODO: Something is still wrong here
        self.regular_grid_max_pts = MAX_NUM_FEATURES

        features = list()
        height = float(self.dh.size[1])
        width = float(self.dh.size[0])
        k = height / width

        n_col = int(np.sqrt(self.regular_grid_max_pts / k))
        n_rows = int(n_col * k)

        h_cols = int(width / n_col)
        h_rows = int(height / n_rows)

        for c in range(n_col):
            for r in range(n_rows):
                features.append(np.array(((c + 0.5) * h_cols, (r + 0.5) * h_rows), dtype=np.float32))

        cpu_f = np.array(features, dtype=np.float32).reshape((1, -1, 2))
        self.cpu_rg = cpu_f

    def do_pnp(self, use_lm_optim=False):
        # TODO: distCoeffs (replace None)
        ret, rvec, tvec = cv2.solvePnP(self.point_cloud, self.cur_c_fts, self.intrinsic_matrix, None, flags=cv2.SOLVEPNP_EPNP)

        if use_lm_optim:
            ret, rvec, tvec = cv2.solvePnP(self.point_cloud, self.cur_c_fts, self.intrinsic_matrix,
                                           None, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

        # Get pose
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.T.reshape(3, 1)

        return R, t

    def draw_fts(self, frame, fts):
        size = 3
        col = (255, 0, 0)
        th = 1
        for f in fts[0]:
            x, y = int(f[0]), int(f[1])
            frame = cv2.circle(frame, (x, y), size, col, thickness=th)
        return frame

    def draw_of(self, frame, pre_fts, cur_fts):
        size = 3
        col = (0, 255, 0)
        th = 2
        for p, c in zip(pre_fts[0], cur_fts[0]):
            end_point = (int(p[0]), int(p[1]))
            start_point = (int(c[0]), int(c[1]))

            frame = cv2.arrowedLine(frame, start_point, end_point, col, th)
        return frame
