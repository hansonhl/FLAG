export CUDA_VISIBLE_DEVICES=$2
python test.py --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1 \
    --data_folder "dataset-$1/" \
    --eval_cluster_number 5 \
    --model_load_path "checkpoints/random-train_10-full_batch_test_final.pth"
# ./run_test.sh {hyperturing1|hyperturing2} 4