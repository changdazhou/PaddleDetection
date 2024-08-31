#!/bin/bash

yaml_list=`ls /paddle/zcd/PaddleDetection/configs/centernet`

for yaml in ${yaml_list[@]};do
    if [[ $yaml == *dla34* ]];then
        yaml_name=${yaml%.*}
        echo ${yaml_name}
        cmd="python -u -m paddle.distributed.launch --gpus='0,1,2,3,4,5,6,7'  --log_dir=./output/${yaml_name}/distributed_train_logs tools/train.py --config configs/centernet/${yaml} --eval "
        echo ${cmd}
    fi
done
