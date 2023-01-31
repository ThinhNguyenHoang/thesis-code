FROM us-central1-docker.pkg.dev/thesis-372616/thesis-registry/base-image:0.1
# RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN sudo rm /usr/share/keyrings/cloud.google.gpg
RUN sudo rm /usr/share/keyrings/cloud.google.gpg~ 

RUN sudo rm /etc/apt/sources.list.d/google-cloud-sdk.list
# RUN wget https://packages.cloud.google.com/apt/doc/apt-key.gpg \
#     && apt-key add apt-key.gpg
RUN wget -O - -q https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y