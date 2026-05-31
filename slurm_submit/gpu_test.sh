#!/bin/bash
#SBATCH --job-name=test-gpu
#SBATCH --partition=mi250x
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=test-%j.out
#SBATCH --mem=4G

echo "HOSTNAME: $(hostname)"

echo $(/opt/rocm/bin/rocminfo | grep "Runtime Version")
echo $(cat /opt/rocm/.info/version)

cd ~/absa-sft-comparison

echo "=== ENV ==="
which python
which pip

echo "=== GPU ENV ==="
env | grep -E 'CUDA|ROCM|HIP|SLURM'

echo "=== ROCm check ==="
which rocminfo
which rocm-smi

echo "=== PYTORCH CHECK ==="

source .venv/bin/activate

python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('hip:', torch.version.hip)
print('device count:', torch.cuda.device_count())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
"