import statistics
from imutils import face_utils
import drawPlots
import camera_flask_app
import dlib
import math
import cv2
import csv
import time
import pandas as pd
from threading import Thread  # library for implementing multi-threaded processing

def write_EAR(data):
    # open csv file
    with open('EAR.csv', 'w', encoding='UTF8', newline="") as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(['Frame Number', 'Average EAR', 'Elapsed Time', 'Status'])
        writer.writerows(data)


######################################################################33
def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dis = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dis


################################# End Function ###############################

########################Aspect Ratio of both eyes######################################
def eyeAspectRatio(Eye):
    # lets calculate individual euclidean distances
    vertical1 = euclidean_distance(Eye[1], Eye[5])
    vertical2 = euclidean_distance(Eye[2], Eye[4])
    horizontal = euclidean_distance(Eye[0], Eye[3])
    EAR = (vertical1 + vertical2) / (2 * horizontal)

    return EAR

def main():
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    panda_EAR = pd.DataFrame(columns=['Frame Number', 'Average EAR', 'Elapsed Time', 'Response'])
    # panda_EAR.to_csv('panda_EAR.csv')

    EAR_data = []
    # EAR_data.to_csv('EAR.csv')

    webcam_stream = cv2.VideoCapture(0)
    # webcam_stream = cv2.VideoCapture(2)

    initial_EAR = []

    initial_mean = math.inf
    initial_sd = math.inf

    original_time = time.perf_counter()  # used to measure time

    ##To measure frame rate
    Num_Frame = 0

    while True:
        # measuring time
        now_time = time.perf_counter()  # used to measure time
        elapsed_time_secs = now_time - original_time

        # Getting out image by webcam
        success, image = webcam_stream.read() 

        if success:

            Num_Frame = Num_Frame + 1  # For frame rate measurement

            # getting height and width
            h, w, _ = image.shape

            # Converting the image to gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Get faces into webcam's image
            rects = detector(gray, 0)

            # Ignore frames which has no detected face
            if len(rects) == 0:
                EAR_data.append((Num_Frame, None, elapsed_time_secs, 'Undefined'))
                # continue

            # For each detected face, find the landmark.
            for (i, rect) in enumerate(rects):
                # Make the prediction and transfom it to numpy array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # we now store the shapes from 36--41 (inclusive) for left eye
                leftEye = [shape[36], shape[37], shape[38], shape[39], shape[40], shape[41]]

                # right eye shapes from 41--47
                rightEye = [shape[42], shape[43], shape[44], shape[45], shape[46], shape[47]]
                left_EAR = eyeAspectRatio(leftEye)
                right_EAR = eyeAspectRatio(rightEye)
                average_EAR = round((left_EAR + right_EAR) / 2, 2)

                if Num_Frame <= 50:
                    initial_EAR.append(average_EAR)
                    cv2.putText(image, 'Training, Try not to blink', (int(w / 2) - 180, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (255, 0, 0), 2, cv2.LINE_AA)
                if Num_Frame == 50:
                    # mean and sd
                    initial_mean = statistics.mean(initial_EAR)
                    initial_sd = statistics.stdev(initial_EAR)
                    # Blink_Status records if there was a blink or not

                Blink_Status = 'Undefined'
                
                if Num_Frame > 50:
                    if average_EAR < initial_mean - 2 * initial_sd:
                        x1 = int(w / 2) - 20
                        cv2.putText(image, 'Blink', (x1, 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        Blink_Status = 'Blink'
                    else:
                        Blink_Status = 'Not Blink'

                EAR_data.append((Num_Frame, average_EAR, elapsed_time_secs, Blink_Status))

                panda_EAR = panda_EAR.append({'Frame Number': Num_Frame, 'Average EAR': average_EAR, 'Elapsed Time': elapsed_time_secs, 'Response': Blink_Status}, ignore_index=True)

                cv2.putText(image, 'Frame= ' + str(Num_Frame), (7, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 255), 1,
                        cv2.LINE_AA)
                cv2.putText(image, 'EAR= ' + str(average_EAR), (w - 125, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 125),
                        1, cv2.LINE_AA)

                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                for (x, y) in shape:
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                
                if not(len(rects) == 0 or len(rects)>1):
                    panda_EAR.to_csv('panda_EAR.csv')
                    
                    # Draw plot
                    rolling = drawPlots.drawPlots()
                    print("After Rolling: ",rolling)

                    if Num_Frame > 200:
                        if rolling < initial_mean - 2 * initial_sd:
                            print("######################################")
                            print("Show alert")
                            # camera_flask_app.drowsiness_alert()
                            break
                        else:
                            print("else:")
                            camera_flask_app.drowsiness_url()

                    # with open('panda_EAR.csv', 'w') as f:
                    #     # create the csv writer
                    #     writer = csv.writer(f)

                    #     # write a row to the csv file
                    #     writer.writerow(panda_EAR)

            try:
                # ret, buffer = cv2.imencode('.jpg', cv2.flip(image,1))
                ret, buffer = cv2.imencode('.jpg', image)
                image = buffer.tobytes()

                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            except Exception as e:
                pass