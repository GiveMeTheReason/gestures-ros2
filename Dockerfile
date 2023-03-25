FROM osrf/ros:foxy-desktop-focal
# FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

ARG ARCH=amd64

RUN apt-get update && apt-get install -y  --allow-unauthenticated \
    apt-utils \
    file \
    dpkg-dev \
    qemu \
    binfmt-support \
    qemu-user-static \
    pkg-config \
    ninja-build \
    doxygen \
    clang \
    python3 \
    python3-pip \
    python3-tk \
    wget \
    gcc \
    g++ \
    git \
    git-lfs \
    nasm \
    cmake \
    curl \
    gpg-agent \
    dbus \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dev:$ARCH \
    libsoundio-dev:$ARCH \
    libjpeg-dev:$ARCH \
    libvulkan-dev:$ARCH \
    libx11-dev:$ARCH \
    libxcursor-dev:$ARCH \
    libxinerama-dev:$ARCH \
    libxrandr-dev:$ARCH \
    libusb-1.0-0-dev:$ARCH \
    libssl-dev:$ARCH \
    libudev-dev:$ARCH \
    mesa-common-dev:$ARCH \
    uuid-dev:$ARCH

RUN wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && rm packages-microsoft-prod.deb && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y libk4a1.4 libk4a1.4-dev k4a-tools

WORKDIR /root/ros_ws/scr
RUN git clone https://github.com/microsoft/Azure_Kinect_ROS_Driver.git -b foxy-devel
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir xacro
RUN apt-get -y install ros-foxy-joint-state-publisher

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    mediapipe

WORKDIR /root/ros_ws
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build

COPY ./azure-kinect/99-k4a.rules /etc/udev/rules.d/

# Update ENTRYPOINT
COPY ./ros_entrypoint.sh /
