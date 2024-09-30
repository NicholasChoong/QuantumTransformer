#!/bin/bash -l
#SBATCH --job-name=tunnel
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=0
#SBATCH --gres=gpu:v100:1
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=21980614@student.uwa.edu.au
#SBATCH --time=3-00:00:00  

# module load Anaconda3/2024.06 cuda/12.4

module list

nvcc --version

nvidia-smi

/usr/sbin/sshd -D -p 2222 -f /dev/null -h ${HOME}/.ssh/id_ecdsa
