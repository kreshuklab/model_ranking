#!/bin/bash

# Script to submit batch jobs for all config files in directory

# Define the config directory
CONFIG_PATH="$1"
PATTERN="${2:-*.yaml}"

if [[ -z "$CONFIG_PATH" ]]; then
    echo "Usage: $0 /path/to/config_dir_or_yaml [pattern]"
    exit 1
fi

if [[ -f "$CONFIG_PATH" ]]; then
    config_files=("$CONFIG_PATH")
elif [[ -d "$CONFIG_PATH" ]]; then
    shopt -s nullglob
    config_files=("$CONFIG_PATH"/$PATTERN)
    shopt -u nullglob
else
    echo "Usage: $0 /path/to/config_dir_or_yaml [pattern]"
    echo "Error: '$CONFIG_PATH' is not a valid file or directory."
    exit 1
fi

if [[ ${#config_files[@]} -eq 0 ]]; then
    echo "No config files found for pattern '$PATTERN' in '$CONFIG_PATH'"
    exit 1
fi

# Define the base directory for SLURM outputs
SLURM_BASE_DIR="../SLURM"

# Create base SLURM directory if it doesn't exist
mkdir -p "$SLURM_BASE_DIR"

# Loop through all meta_config_*.yaml files in the config directory
for config_file in "${config_files[@]}"; do
    
    # Extract the filename without path
    filename=$(basename "$config_file")
    
    # Extract the substring after "meta_config_" and before ".yaml"
    substring="${filename#meta_config_}"
    substring="${substring%.yaml}"
    
    echo "Processing config: $filename"
    echo "Job name substring: $substring"
    
    # Create SLURM output directory for this job
    slurm_dir="$SLURM_BASE_DIR/$substring"
    mkdir -p "$slurm_dir"
    
    # Create a temporary batch script for this specific job
    temp_script="/tmp/batch_job_${substring}.sh"
    
    cat > "$temp_script" << EOF
#! /bin/bash

# this is an example batch script for submitting a gpu job on the cluster
# first, we need to specify some variables for slurm, which is done via the SBATCH comments

#SBATCH -A                              # specify the group
#SBATCH --job-name=$substring                   # specify the name of the job
#SBATCH -N 1				                    # specify the number of cluster nodes for the job
#SBATCH -n 8				                    # specify the number of cores per node for the job
#SBATCH --mem 10G			                    # specify the amount of memory per node
#SBATCH -t 1-00:00:00                           # specify the runtime of the job IMPORTANT: your job will get killed if it exceeds this runtime (the format is d-h:mm-ss)
#SBATCH -o $slurm_dir/outfile.out		        # specify the file to write the command line output to
#SBATCH -e $slurm_dir/errfile.err			    # specify the file to write the error output to
#SBATCH --mail-type=FAIL		            # specify mail notifications for your job 
#SBATCH --mail-user=        # specify the mail address for mail notifications 
#SBATCH -p gpu				                    # specify the queue you want to submit to; here we choose the gpu queue. If you want to submit a pure CPU job, just leave this out.
#SBATCH --gres=gpu:1			                # specify the number of gpus per node

# next we should load all the modules we need to run the job.
# in this example, I just load cuDNN, which pulls in all necessary CUDA dependencies

eval "\$(conda shell.bash hook)"
conda activate /path/to/miniforge3/envs/model-rank-local2

# finally, your script goes here
python batch_prediction_evaluate.py --config $config_file
EOF
    
    # Submit the job
    echo "Submitting job for $substring..."
    sbatch "$temp_script"
    
    # Clean up temporary script
    rm "$temp_script"
    
    echo "Job submitted for $substring"
    echo "---"
done

echo "All jobs submitted!"
