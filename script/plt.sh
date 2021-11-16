python plot_surface_backdoor.py --x=-1:1:51 --y=-1:1:51 --arch $1 --dataset $2 --datapath $3 \
    --model_file $4 --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter \
    --plot 

