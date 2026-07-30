nohup python3 -m torch.distributed.launch --nproc_per_node=4 training/train.py --detector_path ./training/config/detector/sbi.yaml --no-save_ckpt --no-save_feat --ddp > my_output.log 2>&1 &
torchrun --nproc_per_node=4 training/train.py --no-save_ckpt --no-save_feat --ddp
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 training/train.py --no-save_ckpt --no-save_feat --ddp

python -m torch.distributed.launch --nproc_per_node=4 train.py --ddp