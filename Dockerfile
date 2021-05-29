FROM tensorflow/tensorflow:1.14.0-py3

COPY . /pose-estimation-api

WORKDIR /pose-estimation-api

ADD . /pose-estimation-api

RUN python3 -m pip install -r requirements.txt

EXPOSE 5001

CMD ["python3", "main.py"]