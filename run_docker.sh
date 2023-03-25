#!/bin/bash

# allow access from localhost
# DISPLAY=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}'):0
# xhost +$DISPLAY
# xhost +$HOSTNAME.local
# export DISPLAY=:0
xhost +local:docker

# run with X11 forwarding
docker build \
    -f "Dockerfile" \
    -t gestures-ros2:latest "."
docker run \
    --rm \
    --privileged \
    -it \
    --device=/dev/dri \
    --device=/dev/usb/hiddev1 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.Xauthority:/tmp/.Xauthority \
    -v ./mediapipe-extractor/data_samples:/root/ros_ws/src/mediapipe-extractor/data_samples:ro \
    -v ./mediapipe-extractor/scripts:/root/ros_ws/src/mediapipe-extractor/scripts:rw \
    -e XAUTHORITY=/tmp/.Xauthority \
    --env="DISPLAY" \
    gestures-ros2:latest

    # --network=host \
    # -e LIBGL_ALWAYS_INDIRECT=1 \
    # -e DISPLAY=host.docker.internal:0 \
    # -e QT_X11_NO_MITSHM=1 \

# docker-compose up -d --build
# docker run --rm -it -e DISPLAY=$DISPLAY gestures-ros2:latest
