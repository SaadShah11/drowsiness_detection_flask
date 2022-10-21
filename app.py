from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import datetime, time
import os, sys
import numpy as np
import main_landmarks
import main_landmarks_image
import main_webcam
import main_webcam_return
from threading import Thread
import face_utils
import imutils
import dlib

import statistics
from imutils import face_utils
import drawPlots
import dlib
import math
import cv2
import csv
import time
import pandas as pd
import random

global capture, rec_frame, grey, switch, neg, face, rec, out, landmark, video_name, display_alert, drowsiness_val_submitted, drowsiness_value, alert_count
capture = 0
grey = 0
neg = 0
face = 0
switch = 1
rec = 0
landmark = 0
video_name = ''
display_alert = 0
drowsiness_val_submitted = 0
drowsiness_value = -1
alert_count = 0

# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

# make videos directory to save videos
try:
    os.mkdir('./videos')
except OSError as error:
    pass

try:
    os.mkdir('./landmark_videos/videos')
except OSError as error:
    pass

# Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt',
                               './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

# instatiate flask app
app = Flask(__name__, template_folder='./templates')
# app.wsgi_app = ProfilerMiddleware(app.wsgi_app)

camera = cv2.VideoCapture(0)


def record(out):
    global rec_frame
    while (rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame = frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame


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

def random_number():
    # return random.randint(200, 20000)
    return random.randint(1, 400)

def main():
    global display_alert, drowsiness_value, alert_count, drowsiness_val_submitted

    random_alert_frames = [random_number(), random_number(), random_number(), random_number(), random_number()]
    print("============random_alert_frames=============")
    print(random_alert_frames)


    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    panda_EAR = pd.DataFrame(columns=['Frame Number', 'Average EAR', 'Elapsed Time', 'Response', 'Actual Output'])
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
                    cv2.putText(image, 'Training, Try not to blink', (int(w / 2) - 180, h - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
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

                panda_EAR = panda_EAR.append(
                    {'Frame Number': Num_Frame, 'Average EAR': average_EAR, 'Elapsed Time': elapsed_time_secs,
                     'Response': Blink_Status, 'Actual Output': drowsiness_value}, ignore_index=True)

                # Remove the drowsiness_value after adding once
                if drowsiness_value:
                    drowsiness_value = -1

                cv2.putText(image, 'Frame= ' + str(Num_Frame), (7, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 255), 1,
                            cv2.LINE_AA)
                cv2.putText(image, 'EAR= ' + str(average_EAR), (w - 125, 28), cv2.FONT_HERSHEY_SIMPLEX, .7,
                            (0, 255, 125),
                            1, cv2.LINE_AA)

                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                for (x, y) in shape:
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

                if not (len(rects) == 0 or len(rects) > 1):
                    panda_EAR.to_csv('panda_EAR.csv')

                    # Draw plot
                    # rolling = drawPlots.drawPlots()
                    # print("After Rolling: ", rolling)

                    if drowsiness_val_submitted == 1:
                        display_alert = 0
                        drowsiness_val_submitted = 0

                    if Num_Frame > 200:
                        # randomly sending 5 alerts
                        if Num_Frame in random_alert_frames:
                            print("================= Frame Matched ===================")
                            print(Num_Frame)
                            if drowsiness_val_submitted == 0:
                                print("drowsiness_val_submitted: ", drowsiness_val_submitted)
                                display_alert = 1
                            # else:
                            #     display_alert = 0
                            #     drowsiness_val_submitted = 0

                        # if rolling < initial_mean - 2 * initial_sd:
                        #     print("######################################")
                        #     print("Show alert")
                        #     display_alert = 1
                        #     if drowsiness_val_submitted == 0:
                        #         print("drowsiness_val_submitted: ",drowsiness_val_submitted)
                        #         display_alert = 1
                        #     else:
                        #         display_alert = 0
                        #     # break
                        # else:
                        #     print("else:")
                        #     # drowsiness_url()
                        #     if drowsiness_val_submitted == 0:
                        #         print("drowsiness_val_submitted: ",drowsiness_val_submitted)
                        #         display_alert = 1
                        #     else:
                        #         display_alert = 0
                        #     # drowsiness_alert()
                        #     # app.jinja_env.globals.update(drowsiness_alert=drowsiness_alert)

            try:
                # ret, buffer = cv2.imencode('.jpg', cv2.flip(image,1))
                ret, buffer = cv2.imencode('.jpg', image)
                image = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            except Exception as e:
                pass


def main_landmark_image_function(input_image):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

    image = input_image
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 2)

    print("Found {0} Faces!".format(len(rects)))
    try:
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(grey, rect)
            shape = face_utils.shape_to_np(shape)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # show the face number
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    except Exception as e:
        pass
    return image


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame, switch, landmark, display_alert
    while True:
        success, frame = camera.read()
        if success:
            if (face):
                frame = detect_face(frame)
            if (grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (landmark):
                # get landmark image
                # frame = main_landmarks_image.main_landmark_image(frame)
                # frame = main_landmark_image_function(frame)

                # Stopping current webcam
                # print("inside landmarks")
                # switch=0
                # camera.release()
                # cv2.destroyAllWindows()

                # calling landmarks webcam method
                print("landmark enabled")
                # main_webcam.main()

                # landmark = not landmark
            if (display_alert):
                print('display_alert', display_alert)
                # redirect(url_for('drowsiness_alert'))
            if (neg):
                frame = cv2.bitwise_not(frame)
            if (capture):
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)

            if (rec):
                rec_frame = frame
                frame = cv2.putText(cv2.flip(frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 4)
                frame = cv2.flip(frame, 1)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


@app.route('/')
def index():
    return render_template('index.html', display="none")
    # return render_template('index.html', display="none", utc_dt=datetime.datetime.utcnow())


@app.route('/fetch_drowsiness_alert')
def fetch_drowsiness_alert():
    global display_alert
    print("display_alert: ", display_alert)
    return str(display_alert)


@app.route('/fetch_drowsiness_alert_values')
def fetch_drowsiness_alert_values():
    global drowsiness_val_submitted, drowsiness_value
    drowsiness_value = int(request.args.get('drowsiness_value'))
    print("=========================== drowsiness_value =========================")
    print(drowsiness_value)

    if drowsiness_value == 1:
        drowsiness_value = "Extremely Alert"
    elif drowsiness_value == 2:
        drowsiness_value = "Very Alert"
    elif drowsiness_value == 3:
        drowsiness_value = "Alert"
    elif drowsiness_value == 4:
        drowsiness_value = "Fairly Alert"
    elif drowsiness_value == 5:
        drowsiness_value = "Neither Alert Nor Sleepy"
    elif drowsiness_value == 6:
        drowsiness_value = "Some Signs of sleepiness"
    elif drowsiness_value == 7:
        drowsiness_value = "Sleepy, but no effort to keep alert"
    elif drowsiness_value == 8:
        drowsiness_value = "Sleepy, some effort to keep alert"
    elif drowsiness_value == 9:
        drowsiness_value = "Very Sleepy, great effort to keep alert, fighting sleep"
    else:
        drowsiness_value = -1

    print(drowsiness_value)

    drowsiness_val_submitted = 1
    return str(drowsiness_val_submitted)

def drowsiness_alert():
    global display_alert
    # return render_template('index.html', display="block")
    print("display_alert: ", display_alert)
    return 'block' if display_alert == 1 else 'none'


# Add the function by name to the jinja environment.
app.jinja_env.globals.update(drowsiness_alert=drowsiness_alert)


# app.jinja_env.globals['drowsiness_alert'] = drowsiness_alert

def drowsiness_url():
    # with app.test_request_context('/drowsiness_alert'):
    #     print("URL: ", url_for('drowsiness_alert'))
    #     assert request.path == '/drowsiness_alert'
    #     return redirect(url_for('drowsiness_alert'))
    print("redirecting")
    return redirect('/')

    # global display_alert
    # display_alert = 1
    # print('display_alert1', display_alert)
    # gen_frames

    # print("######### URL: ", url_for('drowsiness_alert'))
    # return redirect(url_for('drowsiness_alert'))


@app.route('/video_feed')
def video_feed():
    global switch, landmark

    if (landmark):
        print("inside landmarks")
        switch = 0
        camera.release()
        cv2.destroyAllWindows()

        return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('grey') == 'Grey':
            global grey
            grey = not grey
        elif request.form.get('neg') == 'Negative':
            global neg
            neg = not neg
        elif request.form.get('landmarks') == 'Landmarks':
            print("running main_landmarks")
            # switch=0
            # camera.release()
            # cv2.destroyAllWindows()
            global landmark
            landmark = not landmark
        elif request.form.get('face') == 'Face Only':
            global face
            face = not face
            if (face):
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':
            print("inside switch")
            if (switch == 1):
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out, video_name
            rec = not rec
            if (rec):
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                # out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                # out = cv2.VideoWriter('videos/vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                p = os.path.sep.join(['videos', 'vid_{}.avi'.format(str(now).replace(":", ''))])
                out = cv2.VideoWriter(p, fourcc, 20.0, (640, 480))
                video_name = p
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif (rec == False):
                out.release()
                print("################ Main_Landmarks ##################")
                main_landmarks.main_landmarks(video_name)


    elif request.method == 'GET':
        return render_template('index.html', display="none")
    return render_template('index.html', display="none")


if __name__ == '__main__':
    # app.wsgi_app = ProfilerMiddleware(app.wsgi_app)
    app.run()

camera.release()
cv2.destroyAllWindows()
