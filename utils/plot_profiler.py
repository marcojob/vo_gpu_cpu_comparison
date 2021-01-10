import matplotlib.pyplot as plt
import numpy as np

def main():
    data_dir = "../profiler/backup/"
    datasets = ["KITTI", "Malaga", "Parking", "Autobahn"]
    detectors = ["FAST", "ORB", "SURF", "SHI-TOMASI", "REGULAR_GRID"]
    lims = {"KITTI": [-300, 800, -300, 800],
            "Malaga": [-1300,100, -500, 900],
            "Autobahn": [-1900, 900, -1300, 1500],
            "Parking": [0, 600, -300, 300]}

    for i, a in enumerate(datasets):
        fig = plt.figure(i, figsize=(8, 8), dpi=100)
        for e in detectors:
            filename = data_dir + a + "_" + e + "_1_t.txt"
            x, y = [], []
            with open(filename, "r") as f:
                for line in f.readlines():
                    line = line.rstrip("\n")
                    split = line.split(",")
                    x.append(float(split[0]))
                    y.append(float(split[1]))

            plt.plot(x, y, label=e)

        plt.gca().set_xlim(lims[a][0], lims[a][1])
        plt.gca().set_ylim(lims[a][2], lims[a][3])
        plt.gca().set_ylabel("y [-]")
        plt.gca().set_xlabel("x [-]")
        fig.tight_layout()
        # plt.title("{} dataset".format(a))
        plt.legend()
    
        # Compute avg. std. for data
        for e in detectors:
            filename = data_dir + a + "_" + e + "_0_nfts.txt"
            nfts = list()
            with open(filename, "r") as f:
                for line in f.readlines():
                    n = int(line.rstrip("\n"))
                    nfts.append(n)
            print("{}, {}: {},{}".format(a, e, np.mean(nfts), np.std(nfts)))

    # plt.show()
if __name__ == '__main__':
    main()
