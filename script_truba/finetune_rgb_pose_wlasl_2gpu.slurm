#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A zgokce
#SBATCH -J unisign_rgb_pose
#SBATCH --gres=gpu:2               # Tek node'da 2 GPU
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --time=03-00:00
#SBATCH --output=logs/slurm-%x-%j-%t.out
#SBATCH --error=logs/slurm-%x-%j-%t.err

#export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)


# CUDA bellek yönetimi için fragmentasyonu azalt
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# Dağıtık eğitim için adres ve port
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Resolved master address:"
scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((10000 + RANDOM % 20000))
export NCCL_DEBUG=INFO

source /etc/profile.d/modules.sh
module purge
module load lib/cuda/11.8
module load miniconda3

echo "NODE: $(hostname)"
nvidia-smi

wdir=/arf/home/zgokce/code/Uni-Sign
cd $wdir



conda activate Uni-Sign

output_dir=./out/stage3_finetuning_rgb_pose
ckpt_path=./ckpts/wlasl_rgb_pose_islr.pth

deepspeed ./fine_tuning.py \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset WLASL \
  --task ISLR
  --rgb_support
exit
