import statistics
from imutils import face_utils
import dlib
import math
import cv2
import csv
import time
from threading import Thread  # library for implementing multi-threaded processing

# defining a helper class for implementing multi-threaded processing
class WebcamStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id  # default is 0 for primary camera

        # opening video capture stream
        self.vcap = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is set to False when frames are being read from self.vcap stream
        self.stopped = True

        # reference to the thread for reading next available frame from input stream
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads keep running in the background while the program is executing

    # method for starting the thread for grabbing next available frame in input stream
    def start(self):
        self.stopped = False
        self.t.start()

        # method for reading next frame

    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vcap.read()
            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.vcap.release()

    # method for returning latest read frame
    def read(self):
        return self.frame

    # method called to stop reading frames
    def stop(self):
        self.stopped = True

def main():
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    webcam_stream = WebcamStream(stream_id=0)  # stream_id = 0 is for primary camera
    webcam_stream.start()

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    videoWriter = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    ##To measure frame rate
    Num_Frame = 0

    while True:
        # Getting out image by webcam
        if webcam_stream.stopped is True:
            break
        else:
            image = webcam_stream.read()

        Num_Frame = Num_Frame + 1  # For frame rate measurement

        # getting height and width
        h, w, _ = image.shape
        # Converting the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get faces into webcam's image
        rects = detector(gray, 0)

        if len(rects) == 0:
            continue

        print("outside for")

        # For each detected face, find the landmark.
        for (i, rect) in enumerate(rects):
            # Make the prediction and transfom it to numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


            # videoWriter.write(image)
            cv2.imshow("Output", image)

        # try:
        #     print("inside try")
        #     ret, buffer = cv2.imencode('.jpg', cv2.flip(image,1))
        #     image = buffer.tobytes()
        #     yield (b'--frame\r\n'
        #             b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
        # except Exception as e:
        #     print("inside catch")
        #     pass

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    webcam_stream.stop()  # stop the webcam stream

    cv2.destroyAllWindows()
    videoWriter.release()


if __name__ == "__main__":
    main()
