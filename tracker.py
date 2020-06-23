import argparse

import imutils
from imutils.video import VideoStream
from tensorflow.keras.models import load_model

from utils import *

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--tracker', type = str, required = True,
                help = 'Tracker type: BOOSTING/MIL/KCF/TLD/MEDIANFLOW/MOSSE/CSRT')
ap.add_argument('-c', '--confidence', type = float, default = 0.5,
                help = 'minimum probability to filter weak detections')
args = vars(ap.parse_args())

# Set up tracker.
if args['tracker'] == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
if args['tracker'] == 'MIL':
    tracker = cv2.TrackerMIL_create()
if args['tracker'] == 'KCF':
    tracker = cv2.TrackerKCF_create()
if args['tracker'] == 'TLD':
    tracker = cv2.TrackerTLD_create()
if args['tracker'] == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
if args['tracker'] == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
if args['tracker'] == 'CSRT':
    tracker = cv2.TrackerCSRT_create()

# load our serialized face detector model from disk
print('[INFO] Loading face detector model...')
face_net = cv2.dnn.readNet('deploy.prototxt',
                           'res10_300x300_ssd_iter_140000.caffemodel')

# load the face mask detector model from disk
print('[INFO] Loading face mask detector model...')
mask_net = load_model('mask_detector.model')

# Start video stream
print('[INFO] Starting video stream...')
video = VideoStream(src = 0).start()
# time.sleep(2.0)

# Read first frame and init tracker
frame = video.read()
frame = imutils.resize(frame, width = 1000)
tracker_started = False

while True:
    # Read a new frame
    frame = video.read()
    frame = imutils.resize(frame, width = 1000)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, face_net, mask_net,
                                            args['confidence'])

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (start_x, start_y, end_x, end_y) = box
        (with_mask, with_mask_no_nose, with_mask_under, without_mask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        max_prob = max(with_mask, with_mask_no_nose, with_mask_under,
                       without_mask)
        if max_prob is with_mask:
            label = 'Mask is OK'
            color = (0, 255, 0)  # green
        elif max_prob is with_mask_no_nose:
            label = 'Cover your nose'
            color = (15, 219, 250)  # yellow
        elif max_prob is with_mask_under:
            label = 'Cover yourself'  # orange
            color = (0, 104, 240)
        else:
            label = 'NO mask'
            color = (0, 0, 255)  # red

        # include the probability in the label
        label = '{}: {:.2f}%'.format(label, max_prob * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (start_x, start_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

    # Start timer
    timer = cv2.getTickCount()
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # start tracker on the first face detected
    if not tracker_started and len(locs) != 0:
        tracker_started = True
        bbox = convert_2points_to_1point_and_dims(locs[0])
        track_ok = tracker.init(frame, bbox)

    # Update tracker
    track_ok, bbox = tracker.update(frame)

    # Draw bounding box
    if track_ok:
        # Tracking success
        points = convert_1point_and_dims_to_2points(bbox)
        cv2.rectangle(frame, points[0], points[1], (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, 'Tracking failure detected', (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display tracker type on frame
    cv2.putText(frame, args['tracker'] + ' Tracker', (100, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display FPS on frame
    cv2.putText(frame, 'FPS: ' + str(int(fps)), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display result
    cv2.imshow('Video stream', frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit if ESC pressed
    if key == ord('q'):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
video.stop()
