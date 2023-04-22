FROM datamachines/cudnn_tensorflow_opencv:11.6.2_2.9.1_4.6.0-20220815
COPY ./requirements.txt ./
RUN python -m pip install --upgrade pip 
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8888
CMD jupyter-notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Start the local jupyter for testing with command
# docker run --gpus all -p 8887:8888 -p 6007:6006 -v $(pwd):/dmc local_jupyter_torch_dev:latest