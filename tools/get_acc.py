import os
import re
import matplotlib.pyplot as plt
import os.path as osp

loss_name_dict = {

}

logs_dir = 'output'

def get_value(log_file, loss_name):
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines[::-1]:
            if re.search(loss_name, line) is not None:
                key_str = line.split(loss_name)[1]
                value = float(re.findall(r"\d+\.?\d*",key_str)[0])
                return value

models_loss_dict = {}
for model_logs_dir in os.listdir(logs_dir):
    model_name = model_logs_dir.split('_lr')[0]
    lr = model_logs_dir.split('_lr')[1]
    acc = get_value(osp.join(logs_dir, model_logs_dir,'train.log'), loss_name_dict.get(model_logs_dir,"best metric:"))
    models_info = models_loss_dict.get(model_name, None)
    if models_info is None:
        models_loss_dict[model_name] = {'lr': [lr], 'acc': [acc]}
    else:
        models_loss_dict[model_name]['lr'].append(lr)
        models_loss_dict[model_name]['acc'].append(acc)
    print('model:', model_name, 'lr:', lr, 'acc:', acc)

out_path = 'loss_compare'
if not osp.exists(out_path):
    os.makedirs(out_path)
for model_name in models_loss_dict.keys():
    best_acc = max(models_loss_dict[model_name]['acc'])
    best_acc_index  = models_loss_dict[model_name]['acc'].index(best_acc)
    best_acc_lr = models_loss_dict[model_name]['lr'][best_acc_index]
    print(f"{model_name} best acc is {best_acc}, lr is {best_acc_lr}")
