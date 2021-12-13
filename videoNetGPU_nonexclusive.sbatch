#!/bin/bash
#SBATCH --job-name=gpuvideo
#SBATCH --output=%j_gpuvideo.out
#SBATCH --error=%j_gpuvideo.err
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
##SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --mem=32000

module load python
##source activate tf-1.13.1-p368
module load tensorflow/2.1
##source activate /project2/nicho/weiliang/contractive_autoencoder/code/tf_upgrade_environment
echo "job started at `date`"
##mkdir -p $SCRATCH/$SLURM_JOB_ID
python neural_main.py
##rm -rf $SCRATCH/$SLURM_JOB_ID
echo "job finished at `date`"
