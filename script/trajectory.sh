CUDA_VISIBLE_DEVICES=$1 nohup python plot_trajectory.py \
    --model_folder $2 \
    --start_epoch 0 \
    --max_epoch 99 


