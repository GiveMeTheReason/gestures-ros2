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
from transforms import tt
from classifiers import BaselineClassifier
import torch

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


def landmarks_to_array(landmarks) -> np.ndarray:
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


def points_in_screen(points: np.ndarray) -> np.ndarray:
    return np.prod((0 <= points[:, :2]) * (points[:, :2] <= 1), axis=1).astype(bool)


@tp.overload
def screen_to_pixel(points: np.ndarray, width: int, height: int, inplace: tp.Literal[True]) -> None: ...
@tp.overload
def screen_to_pixel(points: np.ndarray, width: int, height: int, inplace: tp.Literal[False] = False) -> np.ndarray: ...
def screen_to_pixel(
    points: np.ndarray,
    width: int,
    height: int,
    inplace: bool = False,
):
    if not inplace:
        points = np.copy(points)
    points[:, 0] *= width
    points[:, 1] *= height
    if not inplace:
        return points


@tp.overload
def attach_depth(points: np.ndarray, depth_image: np.ndarray, inplace: tp.Literal[True]) -> None: ...
@tp.overload
def attach_depth(points: np.ndarray, depth_image: np.ndarray, inplace: tp.Literal[False] = False) -> np.ndarray: ...
def attach_depth(
    points: np.ndarray,
    depth_image: np.ndarray,
    inplace: bool = False,
):
    if not inplace:
        points = np.copy(points)
    points[:, 2] = depth_image[points[:, 1].astype(int), points[:, 0].astype(int)]
    if not inplace:
        return points


@tp.overload
def pixel_to_world(points: np.ndarray, intrinsic: np.ndarray, inplace: tp.Literal[True]) -> None: ...
@tp.overload
def pixel_to_world(points: np.ndarray, intrinsic: np.ndarray, inplace: tp.Literal[False] = False) -> np.ndarray: ...
def pixel_to_world(
    points: np.ndarray,
    intrinsic: np.ndarray,
    inplace: bool = False,
):
    if not inplace:
        points = np.copy(points)
    focal_x: float = intrinsic[0, 0]
    focal_y: float = intrinsic[1, 1]
    principal_x: float = intrinsic[0, 2]
    principal_y: float = intrinsic[1, 2]
    points[:, 0] = (points[:, 0] - principal_x) * points[:, 2] / focal_x
    points[:, 1] = (points[:, 1] - principal_y) * points[:, 2] / focal_y
    if not inplace:
        return points


def screen_to_world(
    points: np.ndarray,
    depth_image: np.ndarray,
    intrinsic: np.ndarray,
    inplace: bool = False,
):
    if not inplace:
        points = np.copy(points)

    height, width = depth_image.shape
    screen_to_pixel(points, width, height, True)
    attach_depth(points, depth_image, True)
    pixel_to_world(points, intrinsic, True)

    if not inplace:
        return points


def get_world_points(points: np.ndarray, depth_image: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    valid = points_in_screen(points)
    points[~valid] = 0
    points_world = screen_to_world(points, depth_image, intrinsic)
    return points_world


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

        CHECHPOINT_PATH = 'checkpoint.pth'
        GESTURES_SET = (
            'select',
            'call',
            'start',
            'yes',
            'no',
        )
        with_rejection = True
        self.label_map = {gesture: i for i, gesture in enumerate(GESTURES_SET)}
        if with_rejection:
            self.label_map['_rejection'] = len(self.label_map)

        self.inv_map = {value: key for key, value in self.label_map.items()}

        self.transforms = tt
        self.model = BaselineClassifier()
        self.model.load_state_dict(torch.load(CHECHPOINT_PATH, map_location=torch.device('cpu')))
        self.model.eval()

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

        if landmarks.pose_landmarks is not None:
            pixel_points = landmarks_to_array(landmarks.pose_landmarks.landmark)
            filter_visibility(pixel_points, 0.9)
            screen_to_pixel_(pixel_points, self.width, self.height)
            attach_depth_(pixel_points, depth)

            frame_points = landmarks_to_array(landmarks.pose_landmarks.landmark)[:, :3]
            frame_points[25:] = 0.0
            frame_points = get_world_points(frame_points, depth, self.camera_intrinsic)

            with torch.no_grad():
                preds = self.model(self.transforms(frame_points).unsqueeze(0))

            label = self.inv_map[int(torch.argmax(preds).item())]
            cv2.putText(rgb, label, (300, 300),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)

            for point in pixel_points:
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
