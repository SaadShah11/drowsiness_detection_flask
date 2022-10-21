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

global capture, rec_frame, grey, switch, neg, face, rec, out, landmark, video_name, display_alert
capture = 0
grey = 0
neg = 0
face = 0
switch = 1
rec = 0
landmark = 0
video_name = ''
display_alert = 0

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
                redirect(url_for('drowsiness_alert'))
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


@app.route('/drowsiness_alert')
def drowsiness_alert():
    return render_template('index.html', display="block")


def drowsiness_url():
    # redirect(url_for('drowsiness_alert'))
    global display_alert
    display_alert = 1
    print('display_alert1', display_alert)
    gen_frames


@app.route('/video_feed')
def video_feed():
    global switch, landmark

    if (landmark):
        print("inside landmarks")
        switch = 0
        camera.release()
        cv2.destroyAllWindows()

        return Response(main_webcam_return.main(), mimetype='multipart/x-mixed-replace; boundary=frame')
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
                print("################ Video Name ##################")
                print(p)
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif (rec == False):
                out.release()
                print("################ Video Name ##################")
                print(video_name)
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
