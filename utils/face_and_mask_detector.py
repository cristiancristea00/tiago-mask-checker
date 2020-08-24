from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv


class FaceAndMaskDetector:
    """
    Class that encapsulates the functionality of the face and mask detector and
    it's used to get the predictions of the network.
    """

    def __init__(self, confidence):
        """
        Loads the face and mask models and sets the confidence level.
        """
        self.face_net = cv.dnn.readNet('FaceNet.prototxt', 'FaceNet.caffemodel')
        self.mask_net = load_model('mask_detector.model')
        self.confidence = confidence

    def detect_and_predict(self, frame):
        """
        Gets the current frame and returns the predictions and their
        corresponding locations.
        """
        # Grab the dimensions of the frame and then construct a blob from it
        (h, w) = frame.shape[:2]
        blob = cv.dnn.blobFromImage(frame, 1.0, (235, 350))

        # Pass the blob through the network and obtain the face detections
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        # Initialize our list of faces, their corresponding locations, and the
        # list of predictions from our face mask network
        faces = []
        locations = []
        predictions = []

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence associated with the detection
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections by ensuring the confidence is greater
            # than the minimum confidence
            if confidence > self.confidence:
                # Compute the coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype('int')

                # Ensure the bounding boxes fall within the dimensions of
                # the frame
                (start_x, start_y) = (max(0, start_x), max(0, start_y))
                (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

                # Extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and pre-process it
                face = frame[start_y:end_y, start_x:end_x]
                face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
                face = cv.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis = 0)

                # Add the face and bounding boxes to their respective lists
                faces.append(face)
                locations.append((start_x, start_y, end_x, end_y))

        # Only make a predictions if at least one face was detected
        if len(faces) > 0:
            # For faster inference we'll make batch predictions on all faces at
            # the same time rather than one-by-one predictions
            predictions = self.mask_net.predict(faces)

        # Return the face locations and their corresponding locations
        return locations, predictions
