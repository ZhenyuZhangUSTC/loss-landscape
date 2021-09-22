CUDA_VISIBLE_DEVICES=$1 bash script/gpu$1.sh > log_092x_gpu$1.out 2>&1 &
CUDA_VISIBLE_DEVICES=$1 bash script/gpux$1.sh > log_092x_gpux$1.out 2>&1 &
CUDA_VISIBLE_DEVICES=$1 bash script/gpuy$1.sh > log_092x_gpuy$1.out 2>&1 &