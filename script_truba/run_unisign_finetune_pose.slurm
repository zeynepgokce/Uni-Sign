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

source /etc/profile.d/modules.sh
module purge
module load lib/cuda/11.8
module load miniconda3

echo "NODE: $(hostname)"
nvidia-smi

wdir=/arf/home/zgokce/code/Uni-Sign
cd $wdir

export PYTHONPATH=/arf/home/zgokce/miniconda3/envs/open-mmlab/lib/python3.7/site-packages

conda activate Uni-Sign


# DeepSpeed ile eğitimi başlat
deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
  --batch-size 4 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset WLASL \
  --task ISLR
exit
