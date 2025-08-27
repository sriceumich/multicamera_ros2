FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

# Setup environment
ENV DEBIAN_FRONTEND=noninteractive

#########################
# System setup & Locale #
#########################
RUN apt-get update && apt-get install -y curl locales gnupg2 lsb-release

# Set locale
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

ENV ROS_DISTRO=kilted
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ROS 2 kilted sources
RUN mkdir -p /etc/apt/keyrings && \
  curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | gpg --dearmor -o /etc/apt/keyrings/ros.gpg && \
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/ros.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
  > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y \
  python3-colcon-common-extensions \
  python3-flake8 python3-full \
  python3-rosdep python3-dev \
  python3-rosinstall-generator \
  libbullet-dev \
  libasio-dev \
  libtinyxml2-dev \
  libxml2-dev \
  libyaml-cpp-dev \
  gcc-12 g++-12 \
  build-essential cmake git zip \
  libssl-dev libsndfile1-dev ffmpeg libcivetweb-dev \
  libvulkan-dev libgl1-mesa-dev libx11-dev \
  libasound2-dev libpulse-dev libsdl2-dev ccache \
  wget \
  libacl1-dev gcc g++ libncurses5-dev pkg-config \
  dkms \
  python3-pip  python3-transforms3d nano \
  python3-argcomplete \
  libssl-dev bison software-properties-common \
  libglib2.0-dev dpkg\
  kmod udev gcc-12 g++-12 unzip libasound-dev \
  libmysqlcppconn-dev \
  python3-scipy python3-matplotlib \
  python3-requests python3-flask python3-gevent \
  python3-importlib-metadata python3-wheel \
  python3-vcstool python3-empy \
  libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcusparselt0-cuda-12  libcusparselt0-dev-cuda-12

RUN echo "/usr/lib/x86_64-linux-gnu/libcusparseLt/12" | sudo tee /etc/ld.so.conf.d/cusparselt.conf

RUN ldconfig

RUN apt-get update && apt-get install -y \
  qt6-base-dev qt6-base-dev-tools qt6-tools-dev \
  libwebsockets-dev \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-tools \
  gstreamer1.0-libav \
  gstreamer1.0-rtsp  \
  libaravis-0.8-0 \
  libaravis-dev \
  gir1.2-aravis-0.8 aravis-tools \
  aravis-tools-cli  \
  gstreamer1.0-plugins-base-apps \
  x11-apps \
  libx11-6 \
  libx11-xcb1 \
  libxcb1 \
  libxcb-render0 \
  libxcb-shm0 \
  libxcb-xfixes0 \
  libxcb-shape0 \
  libxcb-randr0 \
  libxcb-keysyms1 \
  libxcb-image0 \
  libxcb-icccm4 \
  libxcb-sync1 \
  libxcb-xinerama0 \
  libxcb-xkb1 \
  libxkbcommon-x11-0 \
  libavahi-client-dev \
  libavahi-common-dev \
  vlc \
  nodejs npm \
  libusb-1.0-0 libusb-1.0-0-dev \
  usbutils gdb  \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libjpeg-dev libpng-dev libtiff-dev \
  libavcodec-dev libavformat-dev libswscale-dev

RUN rm -rf /var/lib/apt/lists/* && add-apt-repository universe

#########################
# Build OpenCV 4.10.0   #
#########################
WORKDIR /opt/src
RUN git clone --depth=1 -b 4.10.0 https://github.com/opencv/opencv.git && \
    git clone --depth=1 -b 4.10.0 https://github.com/opencv/opencv_contrib.git && \
    mkdir -p opencv_build
WORKDIR /opt/src/opencv_build

# Python site path for Ubuntu 24.04 (Python 3.12)
ENV PY3_SITE=/usr/local/lib/python3.12/dist-packages
ENV PYTHONNOUSERSITE=1
RUN echo "/usr/local/lib/python3.12/dist-packages" > /usr/local/lib/python3.12/dist-packages/zz-local-first.pth
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV MPLBACKEND=Agg

RUN cmake ../opencv \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D BUILD_LIST=core,highgui,imgproc,videoio,imgcodecs,python3 \
  -D BUILD_opencv_python3=ON \
  -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
  -D PYTHON3_INCLUDE_DIR=/usr/include/python3.12 \
  -D PYTHON3_PACKAGES_PATH=${PY3_SITE} \
  -D OPENCV_PYTHON3_INSTALL_PATH=${PY3_SITE} \
  -D WITH_GSTREAMER=ON \
  -D WITH_FFMPEG=ON

RUN make -j"$(nproc)" && make install && ldconfig

RUN apt-get update && apt-get install -y \
  ros-kilted-ros-base \
  ros-kilted-launch \
  ros-kilted-rclcpp \
  ros-kilted-rclpy \
  ros-dev-tools \
  ros-kilted-desktop \
  ros-kilted-rmw-implementation \
  ros-kilted-std-msgs \
  ros-kilted-iceoryx-hoofs \
  ros-kilted-iceoryx-posh \
  ros-kilted-rviz2 \
  ros-kilted-vision-msgs \
  ros-kilted-cv-bridge \
  ros-kilted-image-transport \
  ros-kilted-rqt \
  ros-kilted-cv-bridge \
  ros-kilted-iceoryx-introspection \
  ros-kilted-depthai-ros \
  ros-kilted-rmw-fastrtps-cpp \
  ros-kilted-desktop \
  ros-kilted-image-pipeline \
  ros-kilted-rqt-common-plugins \
  ros-kilted-rqt-image-view \
  ros-kilted-cv-bridge \
  ros-kilted-image-transport-plugins \
  ros-kilted-depth-image-proc \
  ros-kilted-theora-image-transport \
  ros-kilted-image-proc \
  ros-kilted-image-publisher \
  ros-kilted-image-rotate \
  ros-kilted-stereo-image-proc \
  ros-kilted-compressed-image-transport \
  ros-kilted-compressed-depth-image-transport \
  ros-kilted-camera-calibration \
  ros-kilted-depthai-ros \
  ros-kilted-rmw-cyclonedds-cpp


RUN mkdir -p /etc/udev/rules.d

# Source ROS env
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/kilted/setup.bash" >> ~/.bashrc
RUN echo "source /opt/ros/kilted/setup.bash" >> /etc/bash.bashrc

RUN mkdir -p /etc/udev/rules.d

WORKDIR /home/ubuntu

# Setup DDS config
COPY ./qos/dds_profiles.xml /root/.ros/dds_profilros-dev-toolses.xml

RUN mkdir -p /usr/local/include/nlohmann && \
    curl -L https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp -o /usr/local/include/nlohmann/json.hpp

ENV CUDA_HOME=/usr/local/cuda-12.9
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV CMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc

# Link libcuda stubs to satisfy linker
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1

##############################
# Create group
##############################
# Build arguments for host user and group ID (defaults set to 1000)
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN if getent group ubuntu > /dev/null 2>&1; then \
      groupmod -g ${GROUP_ID} ubuntu; \
    else \
      groupadd --gid ${GROUP_ID} ubuntu; \
    fi

##############################
# Create non-root user
#########################
# 
#  - the UID and GID passed via build arguments,
#  - a home directory,
#  - no log file initialization during account creation (--no-log-init),
#  - bash as the default shell,
#  - membership in the sudo group,
#  - no password (password removed after creation)

RUN if id -u ubuntu > /dev/null 2>&1; then \
      usermod -u ${USER_ID} ubuntu && \
      groupmod -g ${GROUP_ID} ubuntu; \
    else \
      useradd --uid ${USER_ID} --gid ${GROUP_ID} --create-home --no-log-init --shell /bin/bash -G sudo ubuntu; \
    fi && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu && \
    chmod 0440 /etc/sudoers.d/ubuntu && \
    echo "ubuntu:ubuntu" | chpasswd
    
#########################
# Set default user and working directory
#########################
RUN chown ubuntu:ubuntu /home/ubuntu -R
RUN echo "source /home/ubuntu/catkin_ws/install/setup.bash" >> /etc/.bashrc
RUN echo "source /home/ubuntu/catkin_ws/install/setup.bash" >> ~/.bashrc
RUN mkdir -p /data/recordings
RUN chown ubuntu:ubuntu /data -R

#########################
# Python Virtual Env
#########################
USER ubuntu
WORKDIR /home/ubuntu
ARG CACHEBUST=1
WORKDIR  /home/ubuntu/catkin_ws
COPY ./catkin_ws /home/ubuntu/catkin_ws
RUN sudo chown ubuntu:ubuntu /home/ubuntu/catkin_ws -R


# Replace existing CycloneDDS with correct branch
#RUN CYCLONEDDS_BRANCH=$(curl -s https://raw.githubusercontent.com/ros2/ros2/humble/ros2.repos \
#    | grep -A 3 "eclipse-cyclonedds/cyclonedds:" | grep "version:" | awk '{print $2}') && \
#    cd src && \
#    rm -rf eclipse-cyclonedds/cyclonedds && \
#    git clone -b $CYCLONEDDS_BRANCH https://github.com/eclipse-cyclonedds/cyclonedds eclipse-cyclonedds/cyclonedds

# Set CycloneDDS as default RMW
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp


# 1) Remove the wheel that your requirements dropped into the catkin install site
RUN  rm -rf /home/ubuntu/catkin_ws/install/lib/python3.12/site-packages/cv2 \
       /home/ubuntu/catkin_ws/install/lib/python3.12/site-packages/opencv_python* \
       /home/ubuntu/catkin_ws/install/lib/python3.12/site-packages/opencv_contrib_python* \
       /home/ubuntu/catkin_ws/install/lib/python3.12/site-packages/opencv_python_headless*

ENV TORCH_HOME=/home/ubuntu/.cache/torch
RUN mkdir -p /home/ubuntu/.cache/torch/hub/checkpoints
# 1) Download the file (try curl with -L and a UA to avoid 403s)
RUN curl -L -A "Mozilla/5.0" \
  -o /home/ubuntu/.cache/torch/hub/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth

RUN chown -R ubuntu:ubuntu /home/ubuntu/.cache

# 2) Make sure Python prefers /usr/local first
ENV PYTHONPATH=/usr/local/lib/python3.12/dist-packages
ENV PYTHONNOUSERSITE=1 
#RUN pip install -r src/sick_picocam2_bridge/requirements.txt --force-reinstall --ignore-installed  --target /home/ubuntu/catkin_ws/install/lib/python3.12/site-packages

RUN source /opt/ros/kilted/setup.bash && \
             colcon build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                          --merge-install \
                          --event-handlers console_direct+
    
###############################################################################
COPY ./scripts/entrypoint.sh /entrypoint.sh 
COPY ./scripts/identifycamera.py /home/ubuntu/scripts/identifycamera.py
RUN sudo chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
