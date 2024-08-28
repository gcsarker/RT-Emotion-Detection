import math
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

from tensorflow.keras.preprocessing import image


def findEuclideanDistance(source_representation, test_representation):
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def alignment_procedure(img, left_eye, right_eye):
    # this function aligns given face in img based on left and right eye coordinates

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    # -----------------------

    # apply cosine rule

    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    # -----------------------

    return img  # return img anyway

def build_detector_model():
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
    return face_detection


def detect_faces(face_detector, img, align=True):
    detected_face = []

    img_width = img.shape[1]
    img_height = img.shape[0]


    results = face_detector.process(img)

    # If no face has been detected, return an empty list
    if results.detections is None:
        return detected_face

    # Extract the bounding box, the landmarks and the confidence score
    # Taking the first of the faces
    detection = results.detections[0]

    bounding_box = detection.location_data.relative_bounding_box
    landmarks = detection.location_data.relative_keypoints

    x = int(bounding_box.xmin * img_width)
    w = int(bounding_box.width * img_width)
    y = int(bounding_box.ymin * img_height)
    h = int(bounding_box.height * img_height)


    # Extract landmarks
    left_eye = (int(landmarks[0].x * img_width), int(landmarks[0].y * img_height))
    right_eye = (int(landmarks[1].x * img_width), int(landmarks[1].y * img_height))
    

    if x > 0 and y > 0:
        detected_face = img[y : y + h, x : x + w]

        if align:
            detected_face = alignment_procedure(detected_face, left_eye, right_eye)

    #resp.append((detected_face, img_region, confidence))

    bbox = (x,y,w,h)

    return detected_face, bbox



def extract_faces(
    img,
    target_size=(224, 224),
    detector_backend="mediapipe",
    enforce_detection=False,
    align=True,
):
 
    # this is going to store a list of img itself (numpy), it region and confidence
    #extracted_faces = []

    face_detector = build_detector_model()
    #face_objs = detect_faces(face_detector, img, align)
    face, bbox = detect_faces(face_detector, img, align)


    # in case of no face found
    if len(face) == 0 and enforce_detection is True:
        raise ValueError(
            "Face could not be detected. Please confirm that the picture is a face photo "
            + "or consider to set enforce_detection param to False."
        )

    if len(face) == 0 and enforce_detection is False:
        face = img
    
    if face.shape[0] > 0 and face.shape[1] > 0:

        # double check: if target image is not still the same size with target.
        if face.shape[0:2] != target_size:
            face = cv2.resize(face, target_size)

        # normalizing the image pixels
        # what this line doing? must?
        # img_pixels = image.img_to_array(face)
        
        #img_pixels = np.expand_dims(img_pixels, axis=0)
        #img_pixels /= 255  # normalize input in [0, 1]
        

    return face, bbox