import os 
import shutil
import numpy as np 


all_path_list = np.loadtxt('file_path.txt', dtype=np.str)
print(len(all_path_list))
for file_path in all_path_list:
    source_path = os.path.join(file_path[len('mini_cps/'):], 'checkpoint.pth.tar')
    if os.path.exists(source_path):
        print('Exist: ', file_path)
        target_path_head = os.path.join('target_model_0917', file_path[len('mini_cps/'):])
        os.makedirs(target_path_head, exist_ok=True)
        target_path = os.path.join(target_path_head, 'checkpoint.pth.tar')
        shutil.copyfile(source_path, target_path)




