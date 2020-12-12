# Parameters
MAX_NUM_FEATURES = 5000
MIN_NUM_FEATURES = 500
USE_CLAHE = False
MIN_MATCHING_DIFF = 1
PLOT_LIM = 5000

# Shi-Tomasi params
feature_params = dict(maxCorners = 1000,
                      qualityLevel = 0.3,
                      minDistance = 7,
                      blockSize = 7)
