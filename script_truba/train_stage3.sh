output_dir=./out/stage3_finetuning_pose_only

##pose only setting
ckpt_path= ./ckpts/wlasl_pose_only_islr.pth

deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
  --batch-size 4 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset WLASL \
  --task ISLR \
#   --rgb_support # enable RGB-pose setting