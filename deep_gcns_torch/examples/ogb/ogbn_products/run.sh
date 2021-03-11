MACHINE_NAME=$1
export CUDA_VISIBLE_DEVICES=$2
python main.py --use_gpu --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1 \
    --data_folder "dataset-$MACHINE_NAME" \
    --output_folder "log-$MACHINE_NAME"
