import face_utils
import numpy as np
import argparse
import imutils
import dlib
import time
import cv2
import os, sys


def main_landmarks(src):
    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(args["shape_predictor"])
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

    # Reading video
    # input_video = cv2.VideoCapture(args["video"])
    input_video = cv2.VideoCapture(src)
    success, image = input_video.read()
    count = 0

    width = input_video.get(3)  # float `width`
    height = input_video.get(4)  # float `height`

    # This gives total fps!! search more about this
    fps = input_video.get(7)
    print("FPS: ", fps)

    p = os.path.sep.join(['landmark_videos', src])
    print("inside landmarks file")
    print(p)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(p, fourcc, 20.0, (int(width), int(height)))
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(width), int(height)))

    # To measure frame rate
    num_frame = 0

    while success:
        # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
        success, image = input_video.read()

        if success:
            # image = cv2.flip(image, 0)
            # Converting the image to gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Get faces into webcam's image
            rectangles = detector(gray, 2)

            # For each detected face, find the landmark.
            for (i, rect) in enumerate(rectangles):
                # Make the prediction and transfom it to numpy array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                for (x, y) in shape:
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

                cv2.putText(image, 'Frame= ' + str(num_frame), (7, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 255), 1,
                            cv2.LINE_AA)

                out.write(image)
                # cv2.imshow('frame', image)

        print('Reading frame# %d: ', count)
        count += 1

    out.release()
    cv2.destroyAllWindows()


# if __name__ == "__main__":
#     # construct the argument parser and parse the arguments
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-p", "--shape-predictor", required=True,
#                     help="path to facial landmark predictor")
#     ap.add_argument("-v", "--video", required=True,
#                     help="path to input video")
#     args = vars(ap.parse_args())

#     main(args)
