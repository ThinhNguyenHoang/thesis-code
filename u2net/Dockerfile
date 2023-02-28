FROM datamachines/cudnn_tensorflow_opencv:11.6.2_2.9.1_4.6.0-20220815

ENV BUCKET_NAME=thesis-data-bucket
ENV PYTHONDONTWRITEBYTECODE=1

COPY ["./setup.py", "./setup.py"]

RUN pip install --upgrade pip
RUN pip install wget matplotlib
RUN pip install --no-cache-dir .

# Copies the trainer code to the docker image.
COPY ["trainer", "trainer"]


# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train"]