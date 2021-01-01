from argparse import ArgumentParser

from src.vo import VisualOdometry


def main():
    parser = ArgumentParser(description='VO pipeline comparison')
    parser.add_argument("--dataset", "-a",
                        help="Dataset to run pipeline on",
                        type=str,
                        default="KITTI")
    parser.add_argument("--detector", "-e",
                        help="Keypoint detector to use",
                        type=str,
                        default="REGULAR_GRID")
    parser.add_argument("--gpu", "-g",
                        help="Using GPU",
                        type=int,
                        default=1)
    parser.add_argument("--mode", "-m",
                        help="Single or benchmark mode",
                        type=str,
                        default="single")
    parser.add_argument("--headless", "-l",
                        help="Enable headless mode",
                        type=bool,
                        default=False)
    args = parser.parse_args()
    
    vo_gpu = VisualOdometry(args)

if __name__ == '__main__':
    main()
