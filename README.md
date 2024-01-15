# Cirq for QCL

# Setup
```
git clone https://github.com/forest1040/TFQ-docker.git
cd TFQ-docker/cpu
docker build -t tfq .
docker run -it \
  --name qcl tfq:latest bash

pip uninstall numpy
pip install numpy==1.20.3
 ```

