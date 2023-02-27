FROM datamachines/cudnn_tensorflow_opencv:11.6.2_2.9.1_4.6.0-20220815
EXPOSE 8888
CMD jupyter-notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Start the local jupyter for testing with command
# docker run --gpus all -p 8888:8888 -p 6006:6006 -v $(pwd):/dmc jupyter_local_testing:local
# docker run  -p 8888:8888 -p 6006:6006 -v $(pwd):/dmc jupyter_local_testing:local