# FROM us-central1-docker.pkg.dev/thesis-372616/thesis-registry/data_machine_cto:0.1
FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-9:latest

RUN mkdir -m 777 -p /usr/app /home

WORKDIR /usr/app

ENV HOME=/home
ENV BUCKET_NAME=thesis-data-bucket
ENV PYTHONDONTWRITEBYTECODE=1
COPY ["./setup.py", "./setup.py"]
RUN pip install wget python-opencv matplotlib
RUN pip install --no-cache-dir .
# Copies the trainer code to the docker image.
COPY ["trainer", "trainer"]


# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train"]