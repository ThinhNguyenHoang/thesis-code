FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-9:latest

RUN mkdir -m 777 -p /usr/app /home
WORKDIR /home
COPY ./requirements.txt ./


ENV HOME=/home
ENV BUCKET_NAME=thesis-data-bucket
ENV PROJECT_ID=thesis-372616

RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git  -y