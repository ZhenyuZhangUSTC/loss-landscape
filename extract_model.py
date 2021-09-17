import os 
import shutil
import numpy as np 


all_path_list = np.loadtxt('file_path.txt', dtype=np.str)
print(len(all_path_list))
for file_path in all_path_list:
    print(file_path)
    source_path = os.path.join(file_path[len('mini_cps/'):], 'checkpoint.pth.tar')
    target_path = os.path.join('target_model_0917', file_path[len('mini_cps/'):], 'checkpoint.pth.tar')
    shutil.copyfile(source_path, target_path)




