python plot_surface_adv_transfer.py --x=$1:$2:51 --y=$3:$4:51 --model resnet50_adv \
    --model_file $5 \
    --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter \
    --proj_file $6 --dir_file $7 --adv_data --datapath $8
