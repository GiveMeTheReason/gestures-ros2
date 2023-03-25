import mediapipe as mp
import contextlib
import os
import time
import typing as tp

import cv2


def get_file_paths(folder_path: str) -> tp.List[str]:
    dirpath, _, filenames = next(os.walk(folder_path), ('', None, []))
    return [os.path.join(dirpath, filename) for filename in filenames]


PIC_GES = get_file_paths(os.path.join('test_data', 'pictures', 'gestures'))
PIC_POS = get_file_paths(os.path.join('test_data', 'pictures', 'poses'))

VID_GES = get_file_paths(os.path.join('test_data', 'videos', 'gestures'))
VID_POS = get_file_paths(os.path.join('test_data', 'videos', 'poses'))

target_fps = 5


@contextlib.contextmanager
def timer(name: str = ''):
    start = time.time()
    yield
    end = time.time()
    print(f'Code block {name} executed in {end - start} seconds.')


@contextlib.contextmanager
def open_image(path: str):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    yield image
    del image


@contextlib.contextmanager
def capture_video(path: str):
    capture = cv2.VideoCapture(path)
    yield capture
    capture.release()

# --------------------


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def resize_with_aspect_ratio(image, width: tp.Optional[int] = None, height: tp.Optional[int] = None, interpolation=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)

    return cv2.resize(image, dim, interpolation=interpolation)


with timer(), mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5) as pose:
    for filename in PIC_POS:
        with open_image(filename) as image:
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                continue

            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imshow('image', resize_with_aspect_ratio(image, 640))
            cv2.waitKey(0)
cv2.destroyAllWindows()

print([(landmark.x, landmark.y, landmark.z)
      for landmark in results.pose_landmarks.landmark])

with timer(), mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5) as pose:
    for filename in VID_POS:
        with capture_video(filename) as capture:
            frame_rate = capture.get(cv2.CAP_PROP_FPS)
            take_each_frame = (frame_rate + target_fps - 1) // target_fps
            frame_num = -1
            while capture.isOpened():
                ret, image = capture.read()
                if not ret:
                    break

                frame_num += 1
                if frame_num % take_each_frame != 0:
                    continue

                # image_height, image_width, _ = image.shape
                # Convert the BGR image to RGB before processing.
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if not results.pose_landmarks:
                    continue

                # Draw pose landmarks on the image.
                # mp_drawing.draw_landmarks(
                #     image,
                #     results.pose_landmarks,
                #     mp_pose.POSE_CONNECTIONS,
                #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                # cv2.imshow('image', resize_with_aspect_ratio(image))
                # cv2.waitKey(0)
# cv2.destroyAllWindows()
