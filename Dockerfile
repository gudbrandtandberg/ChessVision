FROM python:3.6

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

ADD weights/best_classifier.hdf5 /weights/best_classifier.hdf5
ADD weights/best_extractor.hdf5 /weights/best_extractor.hdf5

COPY ./chessvision/ /code/chessvision
COPY container/container_endpoint_alt.py /container_endpoint.py

ENV CVROOT=/code/chessvision
ENV PYTHONPATH=/code

ENTRYPOINT [ "python" ]
CMD [ "container_endpoint.py" ]

