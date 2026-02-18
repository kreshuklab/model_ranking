#! /bin/bash

# this is an example batch script for submitting a gpu job on the cluster
# first, we need to specify some variables for slurm, which is done via the SBATCH comments

#SBATCH -A kreshuk                              # specify the group
#SBATCH --job-name=RtoR                         # specify the name of the job
#SBATCH -N 1				                    # specify the number of cluster nodes for the job
#SBATCH -n 8				                    # specify the number of cores per node for the job
#SBATCH --mem 10G			                    # specify the amount of memory per node
#SBATCH -t 1-00:00:00                           # specify the runtime of the job IMPORTANT: your job will get killed if it exceeds this runtime (the format is d-h:mm-ss)
#SBATCH -o ../SLURM/RtoR/outfile.out		    # specify the file to write the command line output to
#SBATCH -e ../SLURM/RtoR/errfile.err			# specify the file to write the error output to
#SBATCH --mail-type=FAIL		            # specify mail notifications for your job 
#SBATCH --mail-user=joshua.talks@embl.de        # specify the mail address for mail notifications 
#SBATCH -p gpu				                    # specify the queue you want to submit to; here we choose the gpu queue. If you want to submit a pure CPU job, just leave this out.
#SBATCH --gres=gpu:1			                # specify the number of gpus per node

# next we should load all the modules we need to run the job.
# in this example, I just load cuDNN, which pulls in all necessary CUDA dependencies

eval "$(conda shell.bash hook)"
conda activate /g/kreshuk/talks/miniforge3/envs/model-rank-local2

# finally, your script goes here
python classification_feature_extraction.py --config /g/kreshuk/talks/sampled_features/classification/mitochondria/configs/TTA_configs/RtoR --batch 