#!/bin/bash
#SBATCH --job-name=video
#SBATCH --output=%j_video.out
#SBATCH --error=%j_video.err
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=36:00:00
##SBATCH --mem=32000

module load python
##source activate tf-1.13.1-p368
module load tensorflow/2.1
##source activate /project2/nicho/weiliang/contractive_autoencoder/code/tf_upgrade_environment
echo "job started at `date`"
##mkdir -p $SCRATCH/$SLURM_JOB_ID
python neural_main.py
##rm -rf $SCRATCH/$SLURM_JOB_ID
echo "job finished at `date`"

