#!/usr/bin/env python3

import typing as tp
import numpy as np

import cv2
import mediapipe as mp
import rclpy
from cv_bridge.core import CvBridge
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import CameraInfo, Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


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


def convert_to_array(landmarks) -> np.ndarray:
    result = np.zeros((len(landmarks), 4))
    for i, landmark in enumerate(landmarks):
        result[i] = landmark.x, landmark.y, landmark.z, landmark.visibility
    return result


def filter_visibility(points: np.ndarray, threshold: float):
    # inplace function
    points[points[:, 3] < threshold] = 0.0


def screen_to_pixel_(points: np.ndarray, width: int, height: int):
    # inplace function
    points[:, 0] *= width
    points[:, 1] *= height


def attach_depth_(points: np.ndarray, depth_image: cv2.Mat):
    # inplace function
    points[:, 2] = depth_image[points[:, 1].astype(int), points[:, 0].astype(int)]


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
        self.pose_solver = mp_pose.Pose()

    def _init_camera_info(self, msg: CameraInfo):
        self.width = msg.width
        self.height = msg.height
        self.camera_intrinsic = msg.k.reshape(3, 3)

        self.destroy_subscription(self.camera_info_subscribtion)

    def rgbd_callback(self, rgb_image: Image, depth_image: Image):
        rgb = self.cv_bridge.imgmsg_to_cv2(rgb_image)
        depth = self.cv_bridge.imgmsg_to_cv2(depth_image)
        depth_rgb = cv2.cvtColor((depth / depth.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)

        landmarks = self.pose_solver.process(rgb)

        if landmarks.pose_landmarks is None:
            return

        results = convert_to_array(landmarks.pose_landmarks.landmark)
        filter_visibility(results, 0.9)
        screen_to_pixel_(results, self.width, self.height)
        attach_depth_(results, depth)

        for point in results:
            cv2.circle(rgb, (int(point[0]), int(point[1])), 10, color=(0, 0, 255), thickness=-1)
            cv2.circle(depth_rgb, (int(point[0]), int(point[1])), 10, color=(0, 0, 255), thickness=-1)

        mp_drawing.draw_landmarks(
            rgb,
            landmarks.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

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
