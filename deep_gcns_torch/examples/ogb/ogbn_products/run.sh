export CUDA_VISIBLE_DEVICES=$1
python main.py --use_gpu --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1