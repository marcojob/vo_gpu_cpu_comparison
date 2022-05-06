# Parameters
MAX_NUM_FEATURES = 10000
MIN_NUM_FEATURES = 100
USE_CLAHE = False
MIN_MATCHING_DIFF = 0.1 # 0.002
PLOT_LIM = 5000

# Shi-Tomasi params
feature_params = dict(maxCorners = 1000,
                      qualityLevel = 0.3,
                      minDistance = 7,
                      blockSize = 7)
