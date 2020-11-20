#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --mem=12000
#SBATCH --job-name="STL XAI"
#SBATCH --output=slurm-stl.out
#SBATCH --mail-user=paaledwl@stud.ntnu.no
#SBATCH --mail-type=ALL
 
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"


module purge
module load intel/2020a
module load Python/3.8.2

cd ..
export source ./.env
git pull

type virtualenv
virtualenv env
source env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python main.py stl
deactivate
