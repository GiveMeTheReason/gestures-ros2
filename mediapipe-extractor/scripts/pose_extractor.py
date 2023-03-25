#!/usr/bin/env python3

import typing as tp

import cv2
import mediapipe as mp
import rclpy
from cv_bridge.core import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def resize_with_aspect_ratio(
    image,
    width: tp.Optional[int] = None,
    height: tp.Optional[int] = None,
    interpolation=cv2.INTER_AREA
):
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    dim = (
        width or int(w * height / float(h)),
        height or int(h * width / float(w)),
    )

    resized = cv2.resize(image, dim, interpolation=interpolation)
    return resized


class PoseExtractor(Node):

    def __init__(self):
        super().__init__('pose_detector')

        self.image_subscribtion = self.create_subscription(
            Image,
            'rgb/image_raw',
            self.rgb_callback,
            1,
        )

        self.cv_bridge = CvBridge()
        self.pose = mp_pose.Pose()

    def rgb_callback(self, msg):
        image = self.cv_bridge.imgmsg_to_cv2(msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.pose.process(image)

        print(results.pose_landmarks)

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        cv2.imwrite('/home/player001/gestures_ws/src/mediapipe-extractor/scripts/image.png', image)
        # cv2.imshow('Pose Extractor',
        #            resize_with_aspect_ratio(image, height=600),
        #            )
        # if cv2.waitKey(5) & 0xFF == 27:
        #     rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    pose_extractor = PoseExtractor()

    rclpy.spin(pose_extractor)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
