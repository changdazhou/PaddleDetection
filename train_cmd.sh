yaml_list="DLA34 DLA34_star DLA34_faster"

lr_list="0.05 0.08 0.1 0.2 0.4 0.6 0.8"

config_dir="ppcls/configs/ImageNet/DLA"
base_cmd="python -u -m paddle.distributed.launch --gpus='0,1,2,3,4,5,6,7'"

rm -rf run_list.txt
yaml_files=`ls /paddle/zcd/PaddleDetection/configs/mot/fairmot/fairmot_dla34_30e_1088x608_*`
for yaml_file in ${yaml_files[@]};do
    # yaml_file="configs/mot/fairmot/fairmot_dla34_30e_576x320.yml"
    echo $yaml_file
    yaml_f=${yaml_file##*/}
    yaml_name=${yaml_f%.*}
    # for lr in $lr_list; do
        # output_dir="output/${yaml_name}_lr${lr}"
    cmd="$base_cmd --log_dir=./output/${yaml_name}/distributed_train_logs tools/train.py --config $yaml_file"
    echo $cmd >> run_list.txt
    # done
done