FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel
MAINTAINER Viet Nguyen <nhviet1009@gmail.com>

RUN apt-get update -y && apt-get install -y libglib2.0-dev libsm6 libxext6 libxrender-dev freeglut3-dev ffmpeg
RUN pip install gym-super-mario-bros==7.3.2 opencv-python==4.3.0.36 future==0.18.2 pyglet==1.5.7

WORKDIR /Super-mario-bros-PPO-pytorch
