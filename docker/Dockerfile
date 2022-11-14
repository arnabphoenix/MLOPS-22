# FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
COPY ./serve.py /exp/serve.py
COPY ./model.joblib /exp/./model.joblib
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
EXPOSE 5000
CMD ["python3", "./serve.py"]