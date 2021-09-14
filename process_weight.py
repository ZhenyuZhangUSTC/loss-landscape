import os 
import torch
import sys 
import torchvision.models as models 
import copy 
import torch.nn as nn
import torch.nn.utils.prune as prune

def prune_model_custom(model, mask_dict):

    print('start unstructured pruning with custom mask')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])

def remove_prune(model):
    
    print('remove pruning')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.remove(m,'weight')

def extract_mask(model_dict):

    new_dict = {}

    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = copy.deepcopy(model_dict[key])

    return new_dict

def check_sparsity(model):
    
    sum_list = 0
    zero_sum = 0

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))  

    print('* remain weight = ', 100*(1-zero_sum/sum_list),'%')
    
    return 100*(1-zero_sum/sum_list)

model_file_path = sys.argv[1]
model_file_list = os.listdir(model_file_path)
model_file_list.sort()
for model_path in model_file_list:
    print(model_path)
    model = models.resnet50()
    model_checkpoint = torch.load(os.path.join(model_file_path, model_path), map_location='cpu')
    mask_dict = extract_mask(model_checkpoint)
    prune_model_custom(model, mask_dict)
    model.load_state_dict(model_checkpoint)
    remove_prune(model)
    check_sparsity(model)
    torch.save(model.state_dict(), os.path.join(model_file_path, 'merge_'+model_path))