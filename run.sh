#python S1/train_search.py \
#    --exp_name experiment_name \
#    --m 4\
#    --data_dir ~/.torch/datasets \
#    --seed 2020

python S1/eval_search.py \
    --exp_name search_cifar\
    --m 4\
    --data_dir ~/.torch/datasets \
    --model_path ./super_train/experiment_name/super_train_states.pt.tar\
    --n_generations 200\
    --pop_size 40\
    --n_offsprings 10
