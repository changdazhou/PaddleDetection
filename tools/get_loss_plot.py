import os
import re
from matplotlib import pyplot as plt
from pylab import mpl


log_file = "/paddle/zcd/PaddleDetection/output_1030/fairmot_dla34_30e_1088x608_star_eiou/distributed_train_logs/workerlog.0"

log_info = open(log_file, 'r').readlines()
logs = ";".join(log_info)
heatmap_pat = re.compile('heatmap_loss: (.*) size_loss')
heatmap_loss = heatmap_pat.findall(logs)
size_pat = re.compile('size_loss: (.*) offset_loss')
size_loss = size_pat.findall(logs)
det_pat = re.compile('det_loss: (.*) reid_loss')
det_loss = det_pat.findall(logs)
reid_pat = re.compile('reid_loss: (.*) eta')
reid_loss = reid_pat.findall(logs)
allloss_pat = re.compile('loss: (.*) heatmap_loss')
allloss = allloss_pat.findall(logs)
x_list = list(range(len(heatmap_loss)))
for h_loss,s_loss,d_loss,r_loss,loss in zip(heatmap_loss,size_loss,det_loss,reid_loss,allloss):
    print(f"heatmap_loss:{h_loss};size_loss:{s_loss},det_loss:{d_loss},reid_loss:{r_loss},allloss:{loss}")

plt.rcParams['font.size'] = 24 
plt.figure(figsize=(20, 20))
plt.subplot(2,2,1)
plt.title("heatmap_loss")
plt.xlabel("step")
plt.ylabel("heatmap_loss")
plt.plot(x_list, [float(i) for i in heatmap_loss],linestyle='-', color='b')
plt.subplot(2,2,2)
plt.title("det_loss")
plt.xlabel("step")
plt.ylabel("det_loss")
plt.plot(x_list, [float(i) for i in det_loss],linestyle='-', color='b')
plt.subplot(2,2,3)
plt.title("reid_loss")
plt.xlabel("step")
plt.ylabel("reid_loss")
plt.plot(x_list, [float(i) for i in reid_loss],linestyle='-', color='b')
plt.subplot(2,2,4)
plt.title("total loss")
plt.xlabel("step")
plt.ylabel("loss")
plt.plot(x_list, [float(i) for i in allloss],linestyle='-', color='b')

plt.savefig('plt_images/heatmap_loss.png',bbox_inches='tight', dpi=300)
