#!/usr/bin/env zsh
echo "
cpu_workers: 8

batch_size: 1000
stl_epochs: 40
mtl_epochs: 40

training_size_percentage: 75
test_size_percentage: 15
# Validation size is the rest of the percentage

nodes_before_split: 128
stl_learning_rate: 0.0004
mtl_learning_rate: 0.0004

hidden_layers: [128, 128]
# The activations list needs to be the length of hidden_layers + 1
# as the last layer is provided by nodes_before_split
activations: ['leakyrelu', 'leakyrelu', 'leakyrelu']
# weights or gradients
loss_converge_method: gradients

# data/cchvae_split.py
cchvae_locked_features_heloc: [1, 2, 4]
cchvae_locked_features_gmsc: [2, 10]
" > config.yaml


for noise in {10..50..10}
do
    for i in {1..10}
    do
        python -m data.fake.generateFakeData $noise;
        python -m data.fake.generateOneHotExplanationsFromFake;
        python main.py fake --tag "Exp7-${noise}-MTL-FE" --use_signloss true;
        python main.py fake --tag "Exp7-${noise}-MTL-R";
    done
done

