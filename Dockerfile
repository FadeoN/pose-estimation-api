FROM tensorflow/tensorflow:1.12.0

COPY requirements.txt /

RUN python3 -m pip install -r requirements.txt

COPY . /pose-estimation-api

WORKDIR /pose-estimation-api

ADD . /pose-estimation-api

EXPOSE 5001

CMD ["python3", "main.py"]