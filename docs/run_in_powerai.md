
# **Deployment on IBM Power AI Server**

## Docker Image : We use in this deployed experiment

try to deploy in some environment from ibm power ai base image, here is the list :

| Version     | Docker Image                                     | Python Version | Status |
| ----------- | ------------------------------------------------ | -------------- | ------ |
| 1.5.3       | ibmcom/powerai:1.5.3-all-ubuntu16.04-py3         | 3              |        |
| 1.6.1       | ibmcom/powerai:1.6.1-all-ubuntu18.04-py3         | 3              |        |
| 1.6.2       | ibmcom/powerai:1.6.2-all-ubuntu18.04-py36        | 3.6            |        |
| 1.7.0       | ibmcom/powerai:1.7.0-all-ubuntu18.04-py36        | 3.6.10         | succes |
| 1.7.0       | ibmcom/powerai:1.7.0-all-ubuntu18.04-py37        | 3.7            |        |
| risetai-1.0 | risetai/covid-ppc64le:pneumonia-severity-serving | 3.6.10         | succes |

## Run Docker Image

```
nvidia-docker run -ti --env LICENSE=yes \
--rm -it --name pneumonia_severity \
-p 6663:6663 -p 5001:5001  \
-v /path/to/torchxrayvision/:/workspace/update/torchxrayvision/ \
-v /path/to/pneumonia_severity_serving/:/workspace/pneumonia_severity_serving/ \
-v /pat/to/model-server/:/workspace/model-server/ \
-e NVIDIA_VISIBLE_DEVICES=0,1 \
-w /workspace/ \
DOCKER_IMAGE /bin/bash
```


## Conda : Environment Setting

### Conda Environment Resource
```
https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/linux-ppc64le
https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/noarch
https://repo.anaconda.com/pkgs/main/linux-ppc64le
https://repo.anaconda.com/pkgs/main/noarch
https://repo.anaconda.com/pkgs/free/linux-ppc64le
https://repo.anaconda.com/pkgs/free/noarch
https://repo.anaconda.com/pkgs/r/linux-ppc64le
https://repo.anaconda.com/pkgs/r/noarch
```

### Create Your Conda Environmet form YAML File
```
conda env create -f environment.yml
```

### Install Conda Package
```
conda install package_name
```
### Activate Environment 

```
conda activate environment
```

### Install grpc_tools

```
conda install grpcio-tools
```
### Install Additional Package for torchxray vision

```
python3 -m pip install -r requirement.txt
```


## Install Model Server

come to model-server project directory, then RUN script 'bash create_pip_wheel_and_upload.sh'

add project to your `$PYTHONPATH` 
```
export PYTHONPATH='/workspace/model-server'
```


## Run Your Model : Pneumonia Severity Serving

## Prerequisite 

from setup.py in torchxray vision project,

```
python_requires='>=3.6',
install_requires=[
    'torch>=1',
    'torchvision>=0.5',
    'scikit-image>=0.16',
    'tqdm>=4',
    'numpy>=1',
    'pandas>=1',
    'pydicom>=1',
    'requests>=1'
```

## Run Docker Image

```
nvidia-docker run -ti --env LICENSE=yes \
--rm -it --name pneumonia_severity \
-p 6663:6663 -p 5001:5001  \
-v /path/to/torchxrayvision/:/workspace/update/torchxrayvision/ \
-v /path/to/pneumonia_severity_serving/:/workspace/pneumonia_severity_serving/ \
-v /pat/to/model-server/:/workspace/model-server/ \
-e NVIDIA_VISIBLE_DEVICES=0,1 \
-w /workspace/ \
risetai/covid-ppc64le:pneumonia-severity-serving /bin/bash
```

run serving script inside serving container which use port 5001, 
```
python3 -m model_server.runserver /workspace/pneumonia_severity_serving/pneumonia_severity_servable.py --grpc_port 5001
```


### Testing Seriving Client using Jupyter Notebook
run jupyter notebook inside container, 
```
jupyter notebook --allow-root --no-browser --port 6663 --NotebookApp.token='' --ip 0.0.0.0 
```

acccess client notebook `pneumonia_severity_test.ipynb`  in `/workspace/pneumonia_severity_serving/`,



## References :

* [Docker Hub : powerai](dockehttps://hub.docker.com/r/ibmcom/powerai/)
* [Model Server Library](https://github.com/Abhijit-2592/model-server)
