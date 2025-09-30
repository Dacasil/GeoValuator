#!/bin/bash
#SBATCH --job-name=run_all
#SBATCH -t 08:00:00                  # estimated time
#SBATCH -p grete:shared              # the partition to train on 
#SBATCH -G A100:1                    # take 1 GPU
#SBATCH --mem-per-gpu=8G             # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=8            # number cores per task
#SBATCH --mail-user=l.dacamarasilva@stud.uni-goettingen.de # send mail when job begins and ends
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write slurm output
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate GeoValuator

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env 2> /dev/null

# Print out some git info.
module load git
echo -e "\nCurrent Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Latest Commit: $(git rev-parse --short HEAD)"
echo -e "Uncommitted Changes: $(git status --porcelain | wc -l)\n"

# Run the scripts:
python -u

