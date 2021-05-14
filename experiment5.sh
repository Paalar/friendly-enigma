#!/usr/bin/env zsh
echo "
cpu_workers: 8

batch_size: 1000
stl_epochs: 200
mtl_epochs: 200

training_size_percentage: 75
test_size_percentage: 15
# Validation size is the rest of the percentage

nodes_before_split: 256
stl_learning_rate: 0.0004
mtl_learning_rate: 0.0004

hidden_layers: [256, 256]
# The activations list needs to be the length of hidden_layers + 1
# as the last layer is provided by nodes_before_split
activations: ['leakyrelu', 'leakyrelu', 'leakyrelu']
# weights or gradients
loss_converge_method: gradients

# data/cchvae_split.py
cchvae_locked_features_heloc: [1, 2, 4]
cchvae_locked_features_gmsc: [2, 10]
" > config.yaml

for size in {5..20..5};
    for j in {1..10};
    do
        python main.py partial --module_type mtl-fake --train_size $size --tag "Exp5-${size}P-Statstime";
        python main.py partial --module_type stl-fake --train_size $size --tag "Exp5-${size}P-Statstime";
    done;
done;
