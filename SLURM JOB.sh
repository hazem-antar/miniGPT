#!/bin/bash
#SBATCH --account=
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --gpus-per-node=v100l:4   # Number and type of GPU(s) per node. Check out this link for GPU types in the clusters https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm
#SBATCH --ntasks-per-node=8       # CPU cores/threads
#SBATCH --mem=64G                 # Total memory 
#SBATCH --time=01-00:00           # time (DD-HH:MM)

# Variables
export MAKEFLAGS="-j$(nproc)"
export PYTHONUNBUFFERED=1               # Ensure Python outputs are unbuffered

# Modules for cuda
module load StdEnv/2020
module load gcc/9.3.0
module load cuda/11.4

# Check CUDA version
nvcc --version

# Modules for python
module load python/3.10

# Setup Python environment using virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install required packages
pip install --no-index --upgrade pip

# Install PyTorch from Compute Canada with CUDA 11.4 support
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -f https://download.pytorch.org/whl/cu114/torch_stable.html

# Install Arrow C++ library and set environment variables
module load arrow/13.0.0
export ARROW_HOME=$EBROOTARROW
export LD_LIBRARY_PATH=$ARROW_HOME/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$ARROW_HOME
export ARROW_DIR=$ARROW_HOME

# Install pyarrow using system Arrow libraries
pip install pyarrow==12.0.0 --install-option="--bundle-arrow-cpp" --install-option="--bundle-arrow-cpp-binaries"

# Install compatible versions of transformers and datasets from PyPI
pip install transformers==4.39.3 torchtext==0.14.1 datasets==2.18.0

# Install TorchTriton
pip install triton==2.0.0

# Print device to ensure CUDA is activated
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Start task
python train.py

# Cleaning up
deactivate
