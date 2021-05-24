"""

This is a minimal example of our architecture, training a STL, an MTL and last, an MTL using the full Friendly-Enigma architecture with sign-loss difference.
The config of the networks can be found in config.yaml.
These configurations are not optimized, but should give a short introduction to the system.

Our code is split into "Runners", located in the runners-folder.
These runners contain the configuration, reading of parameters, running the training and finally testing the trained model.

"""

# Setup:
import comet_ml
from utils.minimal_example_helpers import generate_fake_data, print_header, get_auroc_of_model, AUROC_Table

print_header("Setup")

from argparse import ArgumentParser
parser = ArgumentParser(conflict_handler='resolve')
parser.add_argument("--use_signloss", nargs="?", default=False, type=bool)
args = parser.parse_args()

noise_none_table = AUROC_Table("No noise")

# Step 1: Create the synthetic dataset.
# As these are scripts, they are executed on import and does not need to be invoked.
from data.fake import generateFakeData, generateOneHotExplanationsFromFake

# Step 2: STL
# The runners require a seed to log it for reproducability. In this example, we do not use a fixed seed, but add 123 for logging purposes.
print_header("STL")

from runners.STL_Fake_runner import STL_Fake_Runner
runner = STL_Fake_Runner(args=args,seed=123)
runner.run()

stl_no_noise_auroc = get_auroc_of_model(runner)
noise_none_table.append('STL', stl_no_noise_auroc.item())

# Step 3: MTL
print_header("MTL")

from runners.MTL_Fake_runner import MTL_Fake_Runner
runner = MTL_Fake_Runner(args=args, seed=123)
runner.run()

mtl_no_noise_auroc = get_auroc_of_model(runner)
noise_none_table.append('MTL', mtl_no_noise_auroc.item())

# Step 4: MTL with sign-difference loss
print_header("MTL Friendly-Enigma")

parser.add_argument("--use_signloss", default=True)
args = parser.parse_args()

from runners.MTL_Fake_runner import MTL_Fake_Runner
runner = MTL_Fake_Runner(args=args, seed=123)
runner.run()

friendlyenigma_no_noise_auroc = get_auroc_of_model(runner)
noise_none_table.append('MTL-FE', friendlyenigma_no_noise_auroc.item())

# Step 5: STL with on sparse data, 30% noise
print_header("STL with 30% noise")

noise_30_table = AUROC_Table("sparse data, 30% Noise")
generate_fake_data(noise_level=30)

parser.add_argument("--module_type", default="stl-fake")
parser.add_argument("--train_size", default=5)
args = parser.parse_args()

from runners.PartialRunner import PartialRunner
runner = PartialRunner(args=args, seed=123)
runner.run()

stl_30_noise_auroc = get_auroc_of_model(runner)
noise_30_table.append('STL', stl_30_noise_auroc.item())

# Step 6: MTL with 30% noise
print_header("MTL with 30% noise")

parser.add_argument("--use_signloss", default=False)
parser.add_argument("--module_type", default="mtl-fake")
parser.add_argument("--train_size", default=5)
args = parser.parse_args()

runner = MTL_Fake_Runner(args=args, seed=123)
runner.run()

mtl_30_noise_auroc = get_auroc_of_model(runner)
noise_30_table.append('MTL', mtl_30_noise_auroc.item())

# Step 7: MTL Friendly-Enigma with 30% noise
print_header("MTL Friendly-Enigma with 30% noise")

parser.add_argument("--use_signloss", default=True)
parser.add_argument("--module_type", default="mtl-fake")
parser.add_argument("--train_size", default=5)
args = parser.parse_args()

runner = MTL_Fake_Runner(args=args, seed=123)
runner.run()

friendlyenigma_30_noise_auroc = get_auroc_of_model(runner)
noise_30_table.append('MTL-FE', friendlyenigma_30_noise_auroc.item())

# Final: Print AUROC tables for experiments.
print_header("AUROC Tables")
noise_none_table.print()
noise_30_table.print()
