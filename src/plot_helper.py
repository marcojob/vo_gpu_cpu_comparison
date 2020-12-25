import matplotlib.pyplot as plt
import threading

from multiprocessing import Process, Queue
from time import sleep

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

    def plot(self, frame, framerate):
        # This method takes the new data and puts it in the queue
        data = {"frame": frame, "framerate": framerate}
        

        # Put it in the queue
        self.q.put(data)

    def _update(self, q):
        # Data update thread 
        try:
            while True:
                if not q.empty():
                    # Extract data from queue
                    data = q.get()
                    frame = data.get("frame", [])
                    framerate = data.get("framerate", 0.0)

                    # Plot the current frame
                    if not self.current_image_p:
                            self.current_image_p = self.current_image_ax.imshow(frame[...,::-1])
                    else:
                            self.current_image_p.set_data(frame[...,::-1])


                    # Draw the fig again
                    self.fig.canvas.draw_idle()
                
                    # Slow the loop down a bit, this also helps to manage the queue size
                    factor = 1.75
                    if q.qsize() > 100:
                        factor = 1.0
                    sleep(1.0/framerate*factor)
                    print(q.qsize())

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


            # Start the plot update thread
            data_thread = threading.Thread(target=self._update, args=(q,))            
            data_thread.start()


            # Show the plot
            plt.show()

        except Exception as e:
            print(e)

