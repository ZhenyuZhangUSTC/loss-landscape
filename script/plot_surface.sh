python plot_surface_adv_transfer.py --x=$1:$2:31 --y=$3:$4:31 --model resnet50 \
    --model_file loss_landscape_model/$5/merge_99-checkpoint.pth.tar \
    --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter \
    --proj_file loss_landscape_model/$5/PCA/directions.h5_proj_cos.h5 --dir_file loss_landscape_model/$5/PCA/directions.h5 --datapath ../data \
    --plot --surf_file $6



# nips_data1 
# x -10, 10  y -10, 10 
# nips_data10
# x -10 10  y -4, 2
# nips_data100
# x -5 5  y -7, 3



# torch_data1
# x -10 10 y -2, 2
# torch_data10
# x -10 5 y -5, 5
# torch_data100
# x -5 3 y -8 3



# zico_data1
# x -5 10  y -5, 5
# zico_data10
# x -5 5 y -6 2
# zico_data100
# x -10 5 y -2 15


