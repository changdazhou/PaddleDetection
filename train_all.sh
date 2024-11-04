IFS=$'\n'
base_cmd=`cat run_list.txt`


for model_cmd in ${base_cmd[@]};do
    echo $model_cmd
    eval $model_cmd
done