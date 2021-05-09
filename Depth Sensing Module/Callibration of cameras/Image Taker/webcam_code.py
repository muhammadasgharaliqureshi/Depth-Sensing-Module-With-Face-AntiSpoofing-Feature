from threading import Thread
import numpy as np
import cv2


def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

class Camera():

    def __init__(self, srcl, srcr):
        self.capturel = cv2.VideoCapture(srcl)
        self.capturer = cv2.VideoCapture(srcr)

        self.capturel.set(3, 1920 )
        self.capturel.set(4, 1080)

        self.capturer.set(3, 1920)
        self.capturer.set(4, 1080)

        self.framel = np.zeros([520,520,3],dtype= np.uint8)
        self.framer = np.zeros([520,520,3],dtype= np.uint8)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capturel.isOpened():
                (self.status, self.framel) = self.capturel.read()
            #time.sleep(.01)
            if self.capturer.isOpened():
                (self.status, self.framer) = self.capturer.read()

    def capture_frame(self):
        # capture frames in main program
        #framel, framer = rectification(self.framel, self.framer)
        framel = cv2.resize(self.framel, (500,500))
        framer = cv2.resize(self.framer, (500,500))
        #framel = rescale_frame(self.framel, 80)
        #framer = rescale_frame(self.framer, 80)

        #framel = self.framel
        #framer = self.framer
        #print("\n", framer.shape)
        return framel, framer
