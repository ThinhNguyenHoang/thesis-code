FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-9:latest
COPY ['requirements.txt', 'requirements.txt']
RUN pip install requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git  -y