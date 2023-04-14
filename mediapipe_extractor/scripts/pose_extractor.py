#!/usr/bin/env python3

import os
os.environ['PYTHONPATH'] = (
    '/home/player001/gestures_ws/src/mediapipe_extractor/scripts/pose_classifier'
    ':/home/player001/gestures_ws/src/pose_classifier'
    ':/home/player001/gestures_ws/install/message_filters/local/lib/python3.10/dist-packages'
    ':/home/player001/gestures_ws/install/mediapipe_extractor/lib/python3.10/site-packages'
    ':/home/player001/gestures_ws/install/gestures_interfaces/local/lib/python3.10/dist-packages'
    ':/opt/ros/humble/lib/python3.10/site-packages'
    ':/opt/ros/humble/local/lib/python3.10/dist-packages'
)

import typing as tp
import numpy as np

import cv2
import mediapipe as mp
import message_filters
import rclpy
import torch
from cv_bridge.core import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

import kalman_filter
import pose_classifier.visualizer.utils as utils
import pose_classifier.model.classifiers as classifiers
import pose_classifier.model.transforms as transforms


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


CHECHPOINT_PATH = 'checkpoint.pth'
GESTURES_SET = (
    'select',
    'call',
    'start',
    'yes',
    'no',
)
WITH_REJECTION = True


class ModelState:
    def __init__(self) -> None:
        self.hidden_state = None


def resize_with_aspect_ratio(
    image: cv2.Mat,
    target_width: tp.Optional[int] = None,
    target_height: tp.Optional[int] = None,
    interpolation: int = cv2.INTER_AREA
) -> cv2.Mat:
    height, width = image.shape[:2]
    if target_width is None and target_height is None:
        return image
    img_size = (
        target_width or int(width * target_height / float(height)),
        target_height or int(height * target_width / float(width)),
    )

    return cv2.resize(image, img_size, interpolation=interpolation)


def landmarks_to_array(landmarks) -> np.ndarray:
    result = np.zeros((len(landmarks), 4))
    for i, landmark in enumerate(landmarks):
        result[i] = landmark.x, landmark.y, landmark.z, landmark.visibility
    return result


def filter_visibility(points: np.ndarray, threshold: float):
    # inplace function
    points[points[:, 3] < threshold] = 0.0


class PoseExtractor(Node):

    def __init__(self):
        super().__init__('pose_detector')

        self.width: int = 0
        self.height: int = 0
        self.camera_intrinsic = np.eye(1, dtype=np.float64)
        self.camera_info_subscribtion = self.create_subscription(
            CameraInfo,
            'rgb/camera_info',
            self._init_camera_info,
            1,
        )

        image_subscribtion = message_filters.Subscriber(
            self,
            Image,
            'rgb/image_raw',
        )
        depth_subscribtion = message_filters.Subscriber(
            self,
            Image,
            'depth_to_rgb/image_raw',
        )
        self.rgbd_subsribtion = message_filters.ApproximateTimeSynchronizer(
            [image_subscribtion, depth_subscribtion],
            1,
            0.01,
        )
        self.rgbd_subsribtion.registerCallback(self.rgbd_callback)

        self.cv_bridge = CvBridge()
        self.pose_solver = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.label_map = {gesture: i for i, gesture in enumerate(GESTURES_SET, start=1)}
        if WITH_REJECTION:
            # self.label_map['_rejection'] = len(self.label_map)
            self.label_map['_rejection'] = 0

        self.inv_label_map = {value: key for key, value in self.label_map.items()}

        self.transforms = transforms.TestTransforms()
        self.model_state = ModelState()
        self.model = classifiers.LSTMClassifier(len(self.label_map))
        self.model.load_state_dict(torch.load(CHECHPOINT_PATH, map_location=torch.device('cpu')))
        self.model.eval()

    def _init_camera_info(self, msg: CameraInfo):
        self.width = msg.width
        self.height = msg.height
        self.camera_intrinsic = msg.k.reshape(3, 3)

        self.depth_extractor = utils.DepthExtractor(
            self.width,
            self.height,
            self.camera_intrinsic
        )

        self.kalman_filters = kalman_filter.KalmanFilters(
            self.width,
            self.height,
            self.camera_intrinsic,
        )

        self.destroy_subscription(self.camera_info_subscribtion)

    def rgbd_callback(self, rgb_image: Image, depth_image: Image):
        if (self.camera_intrinsic == np.eye(1, dtype=np.float64)).all():
            return

        rgb = self.cv_bridge.imgmsg_to_cv2(rgb_image)
        depth = self.cv_bridge.imgmsg_to_cv2(depth_image)
        depth_rgb = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)

        landmarks = self.pose_solver.process(rgb)

        if landmarks.pose_landmarks is not None:
            mp_points = landmarks_to_array(landmarks.pose_landmarks.landmark)
            frame_points = np.copy(mp_points[:, :3])

            # MP to World
            valid = utils.points_in_screen(frame_points)
            frame_points[~valid] = 0
            self.depth_extractor.screen_to_world(frame_points, depth, True, True)

            # Filtered points
            if self.kalman_filters.filters is None:
                self.kalman_filters.init_filters(frame_points)
                filtered_points = np.copy(frame_points).reshape(-1)
            else:
                filtered_points = self.kalman_filters.make_filtering(
                    mp_points[:, :3],
                    frame_points,
                )

            with torch.no_grad():
                model_points = np.copy(filtered_points[None, ...])
                prediction, self.model_state.hidden_state = self.model(self.transforms(model_points), self.model_state.hidden_state)
                prediction_probs, prediction_label = prediction.max(dim=-1)
                model_label = self.inv_label_map[int(prediction_label.item())]

            cv2.putText(rgb, model_label, (300, 300),
                        cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 8)

            for point in mp_points:
                cv2.circle(rgb, (int(point[0]*self.width), int(point[1]*self.height)), 10, color=(0, 0, 255), thickness=-1)
                cv2.circle(depth_rgb, (int(point[0]*self.width), int(point[1]*self.height)), 10, color=(0, 0, 255), thickness=-1)

            mp_drawing.draw_landmarks(
                rgb,
                landmarks.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        else:
            self.kalman_filters.filters = None

        cv2.imshow('Pose Extractor',
                   resize_with_aspect_ratio(rgb, target_height=600),
        )
        cv2.imshow('Depth',
                   resize_with_aspect_ratio(depth_rgb, target_height=600),
        )

        # cv2.imwrite('/home/player001/gestures_ws/src/image_rgb.png', rgb)
        # cv2.imwrite('/home/player001/gestures_ws/src/image_depth.png', depth_rgb)
        # np.save('/home/player001/gestures_ws/src/image_rgb.npy', rgb)
        # np.save('/home/player001/gestures_ws/src/image_depth.npy', depth)
        if cv2.waitKey(1) & 0xFF == 27:
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    pose_extractor = PoseExtractor()

    rclpy.spin(pose_extractor)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
