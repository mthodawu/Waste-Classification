#docker pull tensorflow/tensorflow:latest-gpu-jupyter  # latest release w/ GPU support and Jupyter

# adds the Project directory to the container on run
docker run -v /home/mtho/Documents/Project:/app/data  -it -p 8888:8888 --runtime=nvidia tensorflow/tensorflow:latest-gpu-jupyter

#docker run --runtime=nvidia -it tensorflow/tensorflow:latest-gpu-jupyter
