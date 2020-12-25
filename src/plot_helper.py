import matplotlib.pyplot as plt
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

    def plot(self, frame, framerate, cloud, nfeatures, framecount):
        # When giving a list into the queue, it actually only gives the pointer
        # therefore you need to create a copy in order to be ensure that content is
        # not modified
        cp_nfeatures = deepcopy(nfeatures)
        
        # This method takes the new data and puts it in the queue
        data = {"frame": frame,
                "framerate": framerate,
                "cloud": cloud,
                "nfeatures": cp_nfeatures,
                "framecount": framecount}
        
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

                    # Draw the fig again
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
                
                    # Slow the loop down a bit, this also helps to manage the queue size
                    factor = 3.0
                    if q.qsize() > 100:
                         factor = 3.00
                    sleep(1.0/framerate*factor)
                else:
                    sleep(1.0)

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
            self.nfeatures_ax.set_title("# tracked landmarks over last 20 frames")
            self.nfeatures_ax.set_ylim(0, 1500)
            self.nfeatures_p = None

            # Start the plot update thread
            data_thread = threading.Thread(target=self._update, args=(q,))            
            data_thread.start()

            # Show the plot
            plt.show()

        except Exception as e:
            print(e)

