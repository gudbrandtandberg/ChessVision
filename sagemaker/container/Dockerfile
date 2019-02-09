# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM tiangolo/uwsgi-nginx-flask:python3.6

# update pip
#RUN apt-get update && apt-get install -y python3-pip

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install all of the packages
RUN pip3 install numpy
RUN pip3 install opencv-python
# opencv-python should be 3.4.1
RUN pip3 install gevent
RUN pip3 install gunicorn
RUN pip3 install keras
RUN pip3 install tensorflow
RUN pip3 install python-chess

# Env Variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV CVROOT=""


# Set up the program in the image
RUN mkdir -p /opt/program/model/
RUN which python3
COPY model/best_extractor.hdf5 /opt/program/model/
COPY model/best_classifier.hdf5 /opt/program/model/
COPY src /opt/program
WORKDIR /opt/program

#ENV LISTEN_PORT=5000
#EXPOSE 5000

#ENTRYPOINT ["python3", "cv_endpoint.py"]
