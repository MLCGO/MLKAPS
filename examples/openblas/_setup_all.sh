#!/bin/bash

python3 -m virtualenv venv
source venv/bin/activate

pushd $(MLKAPS_PATH)/mlkaps
pip install -e .
pip install psutil
pip install seaborn
popd

pushd ./openblas_kernel
mkdir -p build && cd ./build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
popd
