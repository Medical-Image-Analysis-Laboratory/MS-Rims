FROM tensorflow/tensorflow:1.15.3-gpu-py3
RUN mkdir /code
COPY . /code
WORKDIR /code

RUN pip install -r requirements.txt

WORKDIR /code/scripts/inference/
ENTRYPOINT ["python","inference.py"]
  
