from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from utils import *
import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--tracker', type = str, default = 'CSRT', help = 'Tracker type: BOOSTING/MIL/KCF/TLD/MEDIANFLOW/MOSSE/CSRT')
ap.add_argument('-c', '--confidence', type = float, default = 0.5, help = 'Minimum probability to filter weak detections')
ap.add_argument('-T', '--threshold', type = int, default = 60, help = 'Minimum distance between face detection and tracker')
ap.add_argument('-v', '--value', type = int, default = 5, help = 'Number of frames between tracker and detector sync')
ap.add_argument('-w', '--wait', type = int, default = 20, help = 'Number of frames to wait before starting tracker after a face is detected')
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print('[INFO] Loading face detector model...')
face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# load the face mask detector model from disk
print('[INFO] Loading face mask detector model...')
mask_net = load_model('mask_detector.model')

# Start video stream
print('[INFO] Starting video stream...')
video = VideoStream(src = 0).start()

# init tracker
tracker = create_tracker(args['tracker'])
tracker_started = False
counter = args['value']
wait_counter = args['wait']

while True:
    # start timer for FPS
    timer = cv2.getTickCount()

    # Read a new frame
    frame = video.read()
    frame = imutils.resize(frame, width = 1000)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locations, predictions) = detect_and_predict_mask(frame, face_net, mask_net, args['confidence'])

    # loop over the detected face locations and their corresponding
    # locations
    for (box, predic) in zip(locations, predictions):
        # unpack the bounding box and predictions
        (start_x, start_y, end_x, end_y) = box
        (with_mask, with_mask_no_nose, with_mask_under, without_mask) = predic

        # determine the class label and color we'll use to draw
        # the bounding box and text
        max_prob = max(with_mask, with_mask_no_nose, with_mask_under, without_mask)
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
        cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

    # decrement the wait counter for every frame where a face is detected
    if len(locations) != 0 and wait_counter != 0:
        wait_counter -= 1

    # start tracker on the first face detected
    if not tracker_started and len(locations) != 0 and wait_counter == 0:
        tracker_started = True
        bbox = convert_2points_to_1point_and_dims(locations[0])
        track_ok = tracker.init(frame, bbox)

    # Update tracker
    track_ok, bbox = tracker.update(frame)

    # for every frame decrease the counter
    if counter != 0:
        counter -= 1

    # for every set frames, reposition the tracker on the detected face if the
    # distance between their centers are under the threshold value
    if tracker_started and counter == 0 and track_ok:
        counter = args['value']
        tracker_center = get_middle_point(convert_1point_and_dims_to_2points(bbox))
        for box in locations:
            (start_x, start_y, end_x, end_y) = box
            detector_center = get_middle_point(((start_x, start_y), (end_x, end_y)))
            if dist(tracker_center, detector_center) <= args['threshold']:
                bbox = convert_2points_to_1point_and_dims((start_x, start_y, end_x, end_y))
                tracker = create_tracker(args['tracker'])
                track_ok = tracker.init(frame, bbox)
                break

    # Draw bounding box
    if track_ok:
        # Tracking success
        points = convert_1point_and_dims_to_2points(bbox)
        cv2.rectangle(frame, points[0], points[1], (232, 189, 19), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, 'Tracking failure detected!', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Display tracker type on frame
    cv2.putText(frame, args['tracker'] + ' Tracker', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display FPS on frame
    cv2.putText(frame, 'FPS: ' + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Display result
    cv2.imshow('Video stream', frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit if ESC pressed
    if key == ord('q'):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
video.stop()
