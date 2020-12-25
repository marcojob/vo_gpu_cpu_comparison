from src.vo import VisualOdometry

def main():
    dataset =  "Autobahn" #"Malaga" #KITTI"
    vo_gpu = VisualOdometry("gpu", "SURF", dataset)
    vo_gpu.start(plot=True)

if __name__ == '__main__':
    main()
