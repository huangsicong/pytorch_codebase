#!/bin/bash
#SBATCH --partition=t4v1
#SBATCH --job-name=final_vae_train_codebase
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time="-1"
#SBATCH --account=legacy
#SBATCH --qos=legacy
#SBATCH --array=0-2%3
#SBATCH --error=/scratch/ssd001/home/gshensvm/outputs/runerrors.txt

source /h/gshensvm/.bashrc
base_run='cd /scratch/ssd001/home/gshensvm/oodd && python -m codebase.train_gen --hparam_set='
$CUDA_VISIBLE_DEVICES
list=(
"${base_run}dcvae100_mnist"
"${base_run}dcvae100_fmnist"
"${base_run}dcvae100_cifar"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}
