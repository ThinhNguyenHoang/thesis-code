FROM datamachines/jupyter_cto:11.3.1_2.9.1_4.6.0-20220815
EXPOSE 8888
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD jupyter-notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
