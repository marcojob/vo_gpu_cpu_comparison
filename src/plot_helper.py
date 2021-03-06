import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
import threading
import numpy as np

from multiprocessing import Process, Queue
from time import sleep
from copy import deepcopy

"""
    This class spawns a separate plotting process and a data exchange thread, which uses a queue
    in order to exchange data.
    The plotting process runs continuously (but matplotlib plot is blocking). As soon as there
    is data in the queue we take it and update the plot.
"""
class PlotHelper():
    def __init__(self, cpu_or_gpu="GPU"):
         # This method spawns the process and queue
         self.q = Queue()
         self.p = Process(target=self._plot, args=(self.q,))

         # Start the process
         self.p.start()

    def plot(self, frame, framerate, cloud, nfeatures, framecount, trans, color):
        # When giving a list into the queue, it actually only gives the pointer
        # therefore you need to create a copy in order to be ensure that content is
        # not modified
        cp_nfeatures = deepcopy(nfeatures)
        cp_trans = deepcopy(trans)
        
        # This method takes the new data and puts it in the queue
        data = {"frame": frame,
                "framerate": framerate,
                "cloud": cloud,
                "nfeatures": cp_nfeatures,
                "framecount": framecount,
                "trans": cp_trans,
                "color": color}

        # Put it in the queue
        self.q.put(data)

    def _update(self, q):
        # Data update thread 
        try:
            while True:
                if not q.empty():
                    # Extract data from queue
                    data = q.get_nowait()
                    frame = data.get("frame", [])
                    framerate = data.get("framerate", 0.0)
                    cloud = data.get("cloud", np.array([]))
                    nfeatures = data.get("nfeatures", [])
                    framecount = data.get("framecount", None)
                    trans = data.get("trans", [])
                    color = data.get("color", [])

                    # Plot the current frame
                    if not self.current_image_p:
                        self.current_image_p = self.current_image_ax.imshow(frame[...,::-1])
                    else:
                        self.current_image_p.set_data(frame[...,::-1])

                    # Plot the tracked number of features
                    frame_axis = np.linspace(0, len(nfeatures), len(nfeatures))
                    if not self.nfeatures_p:
                        self.nfeatures_p, = self.nfeatures_ax.plot(frame_axis, nfeatures)
                    else:
                        self.nfeatures_p.set_ydata(nfeatures)
                    
                    # Plot the full trajectory
                    trans_x = trans[0]
                    trans_y = trans[1]
                    if not self.full_trajectory_p:
                        self.full_trajectory_p, = self.full_trajectory_ax.plot(trans_x, trans_y)
                    else:
                        ax_min = min(min(trans_x), min(trans_y)) - 10
                        ax_max = max(max(trans_x), max(trans_y)) + 10
                        self.full_trajectory_p.set_xdata(trans_x)
                        self.full_trajectory_p.set_ydata(trans_y)
                        self.full_trajectory_ax.set_xlim(ax_min, ax_max)
                        self.full_trajectory_ax.set_ylim(ax_min, ax_max)

                    # Plot the point cloud, reorganise into easily plottable format
                    cloud_data = [[], []]
                    if not cloud is None and len(cloud.T) > 0:
                        for p in cloud.T:
                            cloud_data[0].append(p[0])
                            cloud_data[1].append(p[2])

                        color = color[:len(cloud.T)]

                        if not self.point_cloud_p:
                            self.point_cloud_p = self.point_cloud_ax.scatter(cloud_data[0], cloud_data[1], c=color, alpha=0.5, s=0.5)
                            self.point_cloud_traj_p, = self.point_cloud_ax.plot(trans_x[-50:], trans_y[-50:])
                        else:
                            ax_x = (trans_x[-1] - 50, trans_x[-1] + 50)
                            ax_y = (trans_y[-1] - 50, trans_y[-1] + 50)
                            self.point_cloud_p.set_offsets(np.array(cloud_data).T)
                            self.point_cloud_p.set_color(np.array(color))
                            self.point_cloud_ax.set_xlim(ax_x)
                            self.point_cloud_ax.set_ylim(ax_y)
                            self.point_cloud_traj_p.set_xdata(trans_x[-50:])
                            self.point_cloud_traj_p.set_ydata(trans_y[-50:])


                    # Draw the fig again
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
                
                    # Slow the loop down a bit, this also helps to manage the queue size
                    factor = 2.0
                    sleep(1.0/framerate*factor)

        except Exception as e:
            print(e)
    
    def _plot(self, q):
        try:
            # Create the figure
            figsize = (10,8)
            dpi = 100
            self.fig = plt.figure(figsize=figsize, dpi=dpi)
            
            # Prepare current image plot axis
            self.current_image_ax = self.fig.add_subplot(2, 2, 1)
            self.current_image_ax.set_title("Current image")
            self.current_image_ax.set_axis_off()
            self.current_image_p = None

            # Prepare the number of tracked features plot
            self.nfeatures_ax = self.fig.add_subplot(2, 4, 5)
            self.nfeatures_ax.set_title("# tracked landmarks over last 50 frames")
            self.nfeatures_ax.set_ylim(0, 2500)
            self.nfeatures_p = None

            # Prepare the full trajectory plot
            self.full_trajectory_ax = self.fig.add_subplot(2, 4, 6)
            self.full_trajectory_ax.set_title("Full trajectory")
            self.full_trajectory_p = None

            # Prepare the point cloud plot
            self.point_cloud_ax = self.fig.add_subplot(1, 2, 2)
            self.point_cloud_ax.set_title("Point cloud and trajectory over last 50 frames")
            self.point_cloud_p = None

            # Start the plot update thread
            data_thread = threading.Thread(target=self._update, args=(q,))            
            data_thread.start()

            # Show the plot
            # figManager = plt.get_current_fig_manager()
            # figManager.window.showMaximized()
            plt.show()

        except Exception as e:
            print(e)

