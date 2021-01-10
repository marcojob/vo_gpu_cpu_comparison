import cv2
import time
import numpy as np
import faulthandler

from src.datasets_helper import DatasetsHelper
from src.plot_helper import PlotHelper
from src.utils import *
from src.profiler import Profiler

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

        # Init plot helper
        if not args.headless:
            self.ph = PlotHelper()

        # Initialize detector
        if self.detector_name == 'FAST':
            if self.on_gpu:
                self.detector = cv2.cuda_FastFeatureDetector.create(threshold=75, nonmaxSuppression=True)
            else:
                self.detector = cv2.FastFeatureDetector_create(threshold=75, nonmaxSuppression=True)
        elif self.detector_name == 'SURF':
            if self.on_gpu:
                self.detector = cv2.cuda.SURF_CUDA_create(300)
            else:
                self.detector = cv2.xfeatures2d.SURF_create(300)
        elif self.detector_name == 'ORB':
            if self.on_gpu:
                self.detector = cv2.cuda_ORB.create(nfeatures=MAX_NUM_FEATURES)
            else:
                self.detector = cv2.ORB_create(nfeatures=MAX_NUM_FEATURES)
        elif self.detector_name == 'SHI-TOMASI':
            if self.on_gpu:
                self.detector = cv2.cuda.createGoodFeaturesToTrackDetector(cv2.CV_8UC1, feature_params['maxCorners'], \
                                            feature_params['qualityLevel'], feature_params['minDistance'], \
                                            feature_params['blockSize'])
            else:
                pass
        elif self.detector_name == 'REGULAR_GRID':
            self.init_regular_grid_detector()
        else:
            assert 1 == 0, "Detector not available"

        # LK
        lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,)
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
        self.cloud = None
        self.fts_color = None

        # Initialize a profiler
        Profiler.set_property_dataset(args.dataset)
        Profiler.set_property_detector(args.detector)
        Profiler.set_property_gpu(args.gpu)

        # Faulthandler
        faulthandler.enable()

        ## Start the pipeline
        self.start(plot= not args.headless)
    
    def start(self, plot=True):
        # Get first frame
        frame = next(self.dh.images)

        # Process first frame
        self.process_first_frame(frame)

        # Loop through frames
        for frame in self.dh.images:
                # Profiler main loop start
                Profiler.start("main_loop")

                # Main frame processing
                self.process_frame(frame)
                self.framecount += 1

                # Profiler main loop end
                Profiler.end("main_loop")

                # Plotting
                if plot:
                    frame = self.cur_rgb_c_frame
                    frame = self.draw_of(frame, self.pre_c_fts, self.cur_c_fts, self.mask_ch)
                
                    self.ph.plot(frame, self.framerate, self.cloud, self.nfeatures_list, self.framecount, self.all_t, self.fts_color) 

        # Generate report
        Profiler.report(self.all_t, self.nfts)

        # Exit afterwards
        exit()

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
        for m, p, c in zip(mask, pre_fts[0], cur_fts[0]):
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
        # Start timer for framerate
        process_frame_start = time.monotonic()
        
        # Upload frame to GPU
        if self.on_gpu:
            Profiler.start("upload_frame")
            self.gf = cv2.cuda_GpuMat()
            self.gf.upload(frame)
            Profiler.end("upload_frame")

        # Resize frame
        if not self.resized_frame_size is None:
            Profiler.start("resize_frame")
            if self.on_gpu:
                self.gf = cv2.cuda.resize(self.gf, self.resized_frame_size)
                self.cur_rgb_c_frame = self.gf.download()
            else:
                self.cur_rgb_c_frame = cv2.resize(frame, self.resized_frame_size)
            Profiler.start("resize_frame")
        else:
            self.cur_rgb_c_frame = frame

        # Convert to gray
        Profiler.start("cvt_color")
        if self.on_gpu:
            self.gf = cv2.cuda.cvtColor(self.gf, cv2.COLOR_BGR2GRAY)
            
            # Update CPU frame
            Profiler.start("download_frame")
            self.pre_c_frame = self.cur_c_frame
            self.cur_c_frame = self.gf.download()
            Profiler.end("download_frame")

        else:
            self.pre_c_frame = self.cur_c_frame
            self.cur_c_frame = cv2.cvtColor(self.cur_rgb_c_frame, cv2.COLOR_BGR2GRAY)
        Profiler.end("cvt_color")

        # Update prev and curr img
        if self.on_gpu:
            self.pre_g_frame = self.cur_g_frame
            self.cur_g_frame = self.gf

        # Detect new features if we don't have enough
        if len(self.pre_c_fts) < MIN_NUM_FEATURES:
            Profiler.start("detect_new_features")
            if self.on_gpu:
                self.cur_g_fts = self.detect_new_features(self.cur_g_frame)
            else:
                self.cur_c_fts = self.detect_new_features(self.cur_c_frame)
            Profiler.end("detect_new_features")
            
            Profiler.start("detect_new_features")
            if self.on_gpu:
                self.pre_g_fts = self.detect_new_features(self.pre_g_frame)
            else:
                self.pre_c_fts = self.detect_new_features(self.pre_c_frame)
            Profiler.end("detect_new_features")

            # Convert keypoints to CPU
            if self.on_gpu:
                Profiler.start("convert_fts_to_cpu")
                self.cur_c_fts = self.convert_fts_gpu_to_cpu(self.cur_g_fts)
                self.pre_c_fts = self.convert_fts_gpu_to_cpu(self.pre_g_fts)
                Profiler.end("convert_fts_to_cpu")

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
        Profiler.start("klt_tracking")
        if self.on_gpu:
            self.pre_c_fts, self.cur_c_fts = self.KLT_tracking(self.pre_g_frame, self.cur_g_frame, self.pre_g_fts)
        else:
            self.pre_c_fts, self.cur_c_fts = self.KLT_tracking(self.pre_c_frame, self.cur_c_frame, self.pre_c_fts)
        Profiler.end("klt_tracking")

        # Reshape features
        self.pre_c_fts = self.pre_c_fts.reshape((1, -1, 2))
        self.cur_c_fts = self.cur_c_fts.reshape((1, -1, 2))
        
        if self.on_gpu:
            # Upload to GPU also
            Profiler.start("upload_frame")
            self.pre_g_fts.upload(self.pre_c_fts)
            self.cur_g_fts.upload(self.cur_c_fts)
            Profiler.end("upload_frame")

        # Find Essential matrix
        Profiler.start("estimate_essential_matrix")
        E, mask = cv2.findEssentialMat(self.cur_c_fts, self.pre_c_fts, self.intrinsic_matrix, cv2.RANSAC, 0.99, 1.0, None)
        Profiler.end("estimate_essential_matrix")

        # Recover pose
        Profiler.start("recover_pose")
        ret, r, t, self.mask_ch = cv2.recoverPose(E, self.cur_c_fts, self.pre_c_fts, self.intrinsic_matrix, mask)
        Profiler.end("recover_pose")

        if ret > 10:
            # Get the color of the feature point, in order to visualize cloud better 
            self.fts_color = list()

            # Only keep mask of features
            tmp_cur_fts = list()
            tmp_pre_fts = list()
            for i, m in enumerate(self.mask_ch):
                if m[0]:
                    tmp_cur_fts.append(self.cur_c_fts[0][i])
                    tmp_pre_fts.append(self.pre_c_fts[0][i])

                    # Get the color value
                    u, v = self.cur_c_fts[0][i]
                    u_i, v_i = int(u), int(i)
                    
                    if u_i >= self.dh.size[0]:
                        u_i = self.dh.size[0] - 1

                    if v_i >= self.dh.size[1]:
                        v_i = self.dh.size[1] - 1

                    self.fts_color.append(self.cur_rgb_c_frame[v_i, u_i, :]/255.0)
            self.cur_c_fts = np.array(tmp_cur_fts).reshape((1, -1, 2))
            self.pre_c_fts = np.array(tmp_pre_fts).reshape((1, -1, 2))

            # Continue tracking of movement
            Profiler.start("track_movement")
            self.scale = 1.0 / np.linalg.norm(t)
            self.cur_r = self.cur_r.dot(r)  # Concatenate the rotation matrix
            self.cur_t = self.cur_t + self.scale * self.cur_r.dot(t)  # Concatenate the translation vectors
            self.all_t[0].append(self.cur_t[0])
            self.all_t[1].append(self.cur_t[2])
            self.all_t[2].append(self.cur_t[1])
            self.all_r.append(self.cur_r)
            Profiler.end("track_movement")

            # Triangulate points
            Profiler.start("triangulate_points")
            cloud_body = self.triangulate_points(r, t)
            Profiler.start("triangulate_points")
            
            # Reject points behind the camera
            Profiler.start("track_pointcloud")
            cloud_body = cloud_body[cloud_body[:, 2] > 0.0]

            # Rotate and translate point cloud
            self.cloud = self.cur_r.dot(cloud_body.T)
            self.cloud = self.cur_t + self.scale*self.cloud
            Profiler.end("track_pointcloud")

            # Update the number of features tracked
            self.nfts.append(len(self.cloud.T))
            self.nfeatures_list.append(len(self.cloud.T))
            if len(self.nfeatures_list) > self.nfeatures_window:
                del self.nfeatures_list[0]
        
        if self.on_gpu:
            # Download frame
            Profiler.start("download_frame")
            self.d_frame = self.gf.download()
            Profiler.end("download_frame")

        # End timer and compute framerate
        framerate = round(1.0 / (time.monotonic() - process_frame_start))
        self.framerate_list.append(framerate) 
        if len(self.framerate_list) > 100:
            del self.framerate_list[0]
        self.framerate = sum(self.framerate_list)/len(self.framerate_list)

    def triangulate_points(self, delta_R, delta_t):
        # Perspective 0 has no rotation and no translation
        P0 = np.dot(self.intrinsic_matrix, np.eye(3, 4))
        
        # Perspective 1 has calculated rotation and translation
        P1 = np.hstack((delta_R, -delta_t))
        P1 = self.intrinsic_matrix.dot(P1)
        
        # Get cloud
        cloud_homogenous = cv2.triangulatePoints(P0, P1, self.pre_c_fts.T, self.cur_c_fts.T)
        cloud = cv2.convertPointsFromHomogeneous(cloud_homogenous.T).reshape(-1, 3)

        return cloud

    def KLT_tracking(self, prev_img, cur_img, prev_fts):
        if self.on_gpu:
            kp2_g, status, error = self.lk.calc(prev_img, cur_img, prev_fts, None)
            kp1_g, status, error = self.lk.calc(cur_img, prev_img, kp2_g, None)

            # Get CPU keypoints
            kp2 = kp2_g.download().reshape((1, -1, 2))
            kp1 = kp1_g.download().reshape((1, -1, 2))
        else:
            kp2, status, error = self.lk.calc(prev_img, cur_img, prev_fts, None)
            kp1, status, error = self.lk.calc(cur_img, prev_img, kp2, None)

        # Find difference
        if self.on_gpu:
            d = abs(prev_fts.download() - kp1).reshape(-1, 2).max(-1)
        else:
            d = abs(prev_fts - kp1).reshape(-1, 2).max(-1)

        diff = d < MIN_MATCHING_DIFF

        # Error Management
        if list(diff).count(True) <= 5:
            return kp1, kp2, MIN_MATCHING_DIFF

        # Create new list with features that fullfil check
        n_kp1, n_kp2 = [], []
        for i, f in enumerate(diff):
            if f:
                n_kp1.append(kp1[0][i])
                n_kp2.append(kp2[0][i])

        # Format features into np array
        n_kp1, n_kp2 = np.array(n_kp1, dtype=np.float32), np.array(
            n_kp2, dtype=np.float32)

        return n_kp1, n_kp2

    def process_first_frame(self, frame):
        # Upload resized frame to GPU
        if self.on_gpu:
            self.gf = cv2.cuda_GpuMat()
            self.gf.upload(frame)

        # Resize frame
        if not self.resized_frame_size is None:
            if self.on_gpu:
                self.gf = cv2.cuda.resize(self.gf, self.resized_frame_size)
                self.cur_rgb_c_frame = self.gf.download()
            else:
                self.cur_rgb_c_frame = cv2.resize(frame, self.resized_frame_size)
        else:
            self.cur_rgb_c_frame = frame

        # Convert to gray
        if self.on_gpu:
            self.gf = cv2.cuda.cvtColor(self.gf, cv2.COLOR_BGR2GRAY)
            
            # Update CPU frame
            self.cur_c_frame = self.gf.download()

            # Update cur image
            self.cur_g_frame = self.gf
        else:
            self.cur_c_frame = cv2.cvtColor(self.cur_rgb_c_frame, cv2.COLOR_RGB2GRAY)

        # Detect initial features
        if self.on_gpu:
            self.cur_g_fts = self.detect_new_features(self.cur_g_frame)

            # Convert keypoints to CPU
            self.cur_c_fts = self.convert_fts_gpu_to_cpu(self.cur_g_fts)
        else:
            self.cur_c_fts = self.detect_new_features(self.cur_c_frame)

        # Track features
        self.pre_c_fts = self.cur_c_fts

        # Reshape
        if self.on_gpu:
            tmp = cv2.cuda_GpuMat()
            tmp_re = self.cur_c_fts.reshape((1, -1, 2))
            tmp.upload(tmp_re)
            self.cur_g_fts = tmp
            self.pre_g_fts = tmp

    def detect_new_features(self, img):
        # Detect features using selected detector
        if self.detector_name == 'FAST' or self.detector_name == 'ORB':
            if self.on_gpu:
                g_kps = self.detector.detectAsync(img, None)
            else:
                c_kps = self.detector.detect(img, None)
                c_kps = np.array([x.pt for x in c_kps], dtype=np.float32).reshape((1, -1, 2))
        elif self.detector_name == 'ORB':
            if self.on_gpu:
                pass
            else:
                c_kps = self.detector.detect(img, None)
                c_kps = np.array([x.pt for x in c_kps], dtype=np.float32).reshape((1, -1, 2))
        elif self.detector_name == 'SURF':
            if self.on_gpu:
                g_kps = self.detector.detect(img, None)
            else:
                c_kps = self.detector.detect(img, None)
                c_kps = np.array([x.pt for x in c_kps], dtype=np.float32).reshape((1, -1, 2))
        elif self.detector_name == 'SHI-TOMASI':
            if self.on_gpu:
                g_kps = self.detector.detect(img)
            else:
                c_kps = cv2.goodFeaturesToTrack(img, **feature_params)
                c_kps = c_kps.reshape((1, -1, 2))
        elif self.detector_name == 'REGULAR_GRID':
            if self.on_gpu:
                g_kps = self.regular_grid_detector()
            else:
                c_kps = self.regular_grid_detector()

        if self.on_gpu:
            return g_kps
        else:
            return c_kps

    def regular_grid_detector(self):
        if self.on_gpu:
            return self.gpu_rg
        else:
            return self.cpu_rg

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
        
        cpu_f = np.array(features, dtype=np.float32).reshape((1, -1, 2))
        if self.on_gpu:
            gpu_f = cv2.cuda_GpuMat()
            gpu_f.upload(cpu_f)
            self.gpu_rg = gpu_f
        else:
            self.cpu_rg = cpu_f

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
