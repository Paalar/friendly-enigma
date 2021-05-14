exit 0;


# TODO
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
cchvae_locked_features_fake: []
" > config.yaml

python -m data.fake.generateFakeData 0;
python -m data.fake.cchvae_split_fake;
mkdir -p cchvae/data/fake;
mv data/fake/fake_*.csv cchvae/data/fake;
cp data/fake/fake_data.csv cchvae/data/fake;
cp checkpoints/stl_fake...nyestedata/last.ckpt cchvae/classifier-fake.ckpt


