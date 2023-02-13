FROM tiangolo/uwsgi-nginx:python3.8

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

ADD weights/best_classifier.hdf5 /weights/best_classifier.hdf5
ADD weights/best_extractor.hdf5 /weights/best_extractor.hdf5

COPY ./chessvision/ /app/chessvision
COPY container/main.py /app/main.py
COPY container/uwsgi.ini /app/uwsgi.ini

ENV CVROOT=/app/chessvision
ENV PYTHONPATH=/app

ENV LISTEN_PORT 8080
EXPOSE 8080

ENV UWSGI_CHEAPER 0
ENV UWSGI_PROCESSES 1

