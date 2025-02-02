python plot_surface_adv_transfer.py --x=$1:$2:31 --y=$3:$4:31 --model resnet50 \
    --model_file loss_landscape_model/$5/merge_99-checkpoint.pth.tar \
    --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter \
    --proj_file loss_landscape_model/$5/PCA/directions.h5_proj_cos.h5 --dir_file loss_landscape_model/$5/PCA/directions.h5 --datapath ../data \
    --plot --surf_file $6



# nips_data1 
# x -10, 50  y -15, 10 
# nips_data10
# x -10 10  y -10, 10
# nips_data100
# x -20 10  y -10, 10



# torch_data1
# x -10 20 y -10, 10
# torch_data10
# x -15 20 y -5, 5
# torch_data100
# x -20 10 y -20 10



# zico_data1
# x -5 25  y -5, 20
# zico_data10
# x -20 10 y -10 6
# zico_data100
# x -20 10 y -8 22


