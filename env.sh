#!/bin/bash

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n MLPhys python=3.8 -y
conda activate MLPhys
pip install --upgrade pip
echo "Installing TensorFlow (v2.13.0) and TensorFlow Probability..."
pip install tensorflow==2.13.0 tensorflow_probability
pip install pyDOE
conda install -c conda-forge numpy matplotlib -y

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "MLPhys env"