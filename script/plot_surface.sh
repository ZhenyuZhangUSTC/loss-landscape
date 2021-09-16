python plot_surface_adv_transfer.py --x=$1:$2:51 --y=$3:$4:51 --model resnet50 \
    --model_file loss_landscape_model/$5/merge_99-checkpoint.pth.tar \
    --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter \
    --proj_file loss_landscape_model/$5/PCA/directions.h5_proj_cos.h5 --dir_file loss_landscape_model/$5/PCA/directions.h5 --datapath $6



# nips_data1 
# x -10, 45  y -2, 10 
# nips_data10
# x -10 50  y -6, 4
# nips_data100
# x -20 90  y -20, 10



# torch_data1
# x -10 50 y -5, 10
# torch_data10
# x -20 70 y -10, 5
# torch_data100
# x -20 90 y -20 10



# zico_data1
# x -5 25  y -8, 20
# zico_data10
# x -20 130 y -10 8
# zico_data100
# x -30 160 y -10 25


