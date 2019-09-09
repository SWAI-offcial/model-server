#!/bin/shell

COUNT=$(docker ps|grep tensorflow/serving |grep -v 'grep' |wc -l)
echo $COUNT
if [ $COUNT -eq 0 ];then
    echo "Start docker tensorflow/serving east"
    docker run -p 8501:8501 --mount type=bind,source=/home/guoningyan/data/east,target=/models/east -e MODEL_NAME=east -t tensorflow/serving &
else
    echo "docker tensorflow/serving east running"
fi

source activate taxt
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
nohup python app.py &
