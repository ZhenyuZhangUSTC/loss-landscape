mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51 \
--mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter \
--ticket_dir $1 --model_file $2


