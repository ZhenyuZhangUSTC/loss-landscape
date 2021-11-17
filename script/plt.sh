CUDA_VISIBLE_DEVICES=$5 python plot_surface_backdoor.py --x=-1:1:51 --y=-1:1:51 --arch $1 --dataset $2 --data $3 \
    --model_file $4 --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter \
    --plot --upper_right --seed $6 --surf_file $7 