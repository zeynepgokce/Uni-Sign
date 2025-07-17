#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A zgokce
#SBATCH -J unisign_1gpu
#SBATCH --gres=gpu:1                # Tek node'da 1 GPU
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=03-00:00
#SBATCH --output=logs/slurm-%x-%j-%t.out
#SBATCH --error=logs/slurm-%x-%j-%t.err


export MASTER_PORT=$((12000 + $RANDOM % 20000))
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)


source /etc/profile.d/modules.sh
module purge
module load lib/cuda/11.8
module load miniconda3

echo "NODE: $(hostname)"
nvidia-smi

wdir=/arf/home/zgokce/code/Uni-Sign
cd $wdir



conda activate Uni-Sign

output_dir=./out/stage3_finetuning_pose_only
ckpt_path=./ckpts/wlasl_pose_only_islr.pth

srun deepspeed ./fine_tuning.py \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset WLASL \
  --task ISLR
exit
