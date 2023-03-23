# Docker for metavision tools
This docker is intended to allow you to use several event camera related tools, in particular the metavision sdk
 by [Prophesee](https://docs.prophesee.ai/stable/metavision_sdk/).

## Installation

To use this you will need to install [Docker](https://www.docker.com/) and [Docker-compose](https://docs.docker.com/compose/)
 
To use the machine learning samples you will need to install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker):
 
 ```bash
 distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
```shell
sudo apt-get update
```
```bash
sudo apt-get install -y nvidia-container-toolkit
```
```bash
sudo nvidia-ctk runtime configure --runtime=docker
```
```bash
sudo systemctl restart docker
 ```
 
 ## Run
 You can build the docker using docker-compose by doing:
 ```
docker-compose up --d
```
The docker compose file is intended to host several submodules. To rebuild a submodule (e.g. metavision_sdk) do:
```
docker-compose -f docker-compose.yml build metavision_sdk
```
You can get inside the docker in an interactive shell by doing (the example refers to metavision_sdk, change it if you want to access different images):
```
docker-compose exec metavision_sdk bash
```
The metavision_sdk dockerfile comes with VNC support. To run it, enter the interactive shell with the previous command and run:
```
/data/run_vnc.sh
```
Then use your VNC client to connect to the docker using localhost:5920 (you can change this in the metavision_sdk Dockerfile) and password "TestVNC" (you can change this in "run_vnc.sh")

## Detection example
To run a detection sample from the metavision sdk you will need to download a model and sample from [metavision](https://docs.prophesee.ai/stable/metavision_sdk/modules/ml/samples/detection_and_tracking_inference.html#chapter-sdk-ml-detection-and-tracking-inference)
You will find a script to launch the detection_and_tracking_pipeline sample, which use the "red_histogram_05_2020" model and the "driving_sample.raw" video.
You will first need to activate VNC as explained above, then run from the interactive docker shell:
```
/data/metavision_detection_sample/run_detection.sh
```
This should show detected vehicles in the VNC window.
