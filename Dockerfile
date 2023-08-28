## For running this we need the container to have access to the host's network, so run:
## docker run -it --network host {image_name}
## If you want display run:
## docker run -it --network host -e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix {image_name}

FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.13-py3
## Setting env variables
ARG username
ENV USER_NAME $username
ARG password
ENV PASSWORD $password
ARG repo_url
ENV REPO_URL $repo_url

## Set up the working directory
RUN mkdir /app
WORKDIR /app

## Cloning the repository
RUN git config --global user.name $USER_NAME
RUN git config --global user.password $PASSWORD
RUN git clone https://$PASSWORD@$REPO_URL

## Install GStreamer and Python GStreamer bindings
RUN apt-get update && apt-get install -y \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    python3-gst-1.0 \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-3.0 \
    gir1.2-gst-plugins-base-1.0 \
    gir1.2-gstreamer-1.0

WORKDIR ./DeepSortTracking

## Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

## Install gstreamer-python (without the PyCairo version requirement)
RUN pip3 install git+https://github.com/jackersson/gstreamer-python.git --no-dependencies


## Copy the network weights
RUN mkdir /app/DeepSortTracking/weights
COPY weights /app/DeepSortTracking/weights

# Exposing ports to listen
EXPOSE 5004

