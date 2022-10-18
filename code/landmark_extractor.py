import cv2
import dlib
import numpy as np

def get_grayscale(image):
  # Channel is 2 or more
  if image.shape[-1] > 1:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  else:
    gray = image.astype(np.uint8)
  return gray

def get_detected_faces(gray_image):
  # input is a grayscale image
  # this function return a list of detected faces
  # using dlib face detector
  detector = dlib.get_frontal_face_detector()
  detected_faces = detector(gray_image, 1)
  return detected_faces

def get_landmarks(gray_image, gray_face):
  # input is a grayscale face
  # using dlib shape detector with face landmarks input
  path = './checkpoints/shape_predictor_68_face_landmarks.dat'
  predictor = dlib.shape_predictor(path)
  landmarks = predictor(gray_image, gray_face)
  return landmarks

def generate_part(part_arr_idx, landmarks):
  part = []
  for n in part_arr_idx:
    x = landmarks.part(n - 1).x
    y = landmarks.part(n - 1).y
    temp = np.array([x, y], np.float32)
    part.append(temp)
  part = np.array(part, dtype=np.int32)
  return part

def create_mask(image):
  mask = np.zeros_like(image)
  gray_image = get_grayscale(image)

  detected_faces = get_detected_faces(gray_image)

  ## no face detected
  if len(detected_faces) == 0:
    return mask, False

  for face in detected_faces:
    # there is just one face in each image of FER-2013
    # so this loop will be executed once
    landmarks = get_landmarks(gray_image, face)

  # Generate left eye part
  left_eye_idx = range(37, 43)
  left_eye = generate_part(left_eye_idx, landmarks)

  # Generate right eye part
  right_eye_idx = range(43, 49)
  right_eye = generate_part(right_eye_idx, landmarks)

  # Generate nose part
  nose_idx = [28, 32, 33, 34, 35, 36]
  nose = generate_part(nose_idx, landmarks)

  # Generate mouth part
  mouth_idx = range(49, 61)
  mouth = generate_part(mouth_idx, landmarks)

  contours = [left_eye, right_eye, nose, mouth]
  for cnt in contours:
    for x in range(mask.shape[0]):
      for y in range(mask.shape[1]):
        ret = cv2.pointPolygonTest(cnt, (y, x), False)
        if ret >= 0:
            mask[x, y, :] = image[x, y, :]
  
  return mask, True

def create_mask_wrapper(image):
  mask, flag = create_mask(image)
  return mask