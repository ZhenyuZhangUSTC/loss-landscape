python plot_surface_adv_transfer.py --x=$1:$2:31 --y=$3:$4:31 --model resnet50 \
    --model_file loss_landscape_model/$5/merge_99-checkpoint.pth.tar \
    --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter \
    --proj_file loss_landscape_model/$5/PCA/directions.h5_proj_cos.h5 --dir_file loss_landscape_model/$5/PCA/directions.h5 --datapath loss_landscape_model/$5 --adv_data
