CUDA_VISIBLE_DEVICES=$1 bash script/gpu$1.sh > log_0921_gpu$1.out 2>&1 &
CUDA_VISIBLE_DEVICES=$1 bash script/gpux$1.sh > log_0921_gpux$1.out 2>&1 &