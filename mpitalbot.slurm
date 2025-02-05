#!/bin/bash
#SBATCH --job-name=talbot_job     # Job name
#SBATCH --output=talbot_job%j.txt # Standard output and error log (%j is the job ID)

#SBATCH --nodes=12
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB                 # Total memory for the job

#SBATCH --time=5-00:00:00           # Maximum time for the job (HH:MM:SS)
#SBATCH --partition=large        # Specify the partition

# Print information about the job and the node
echo "Starting SLURM job..."
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Step 1: Load necessary modules (modify based on system configuration)
module load Anaconda3/py010822          # Load the Anaconda module if it's available
module load GSL/2.7-GCC-10.3.0          # Load the GSL module (check availability)
module load FFmpeg/4.3.2-GCCcore-10.3.0 # Load the FFmpeg module (check availability)
module load OpenMPI/4.0.5-GCC-10.2.0    # Load OpenMPI

# Enter conda
conda activate talbotenv

# Step 2: Run Makefile (compiling code using make)
rm pyTalbot/pyTalbot/src/*.so
echo "Running Makefile..."
make -C pyTalbot/pyTalbot/src/  # Run the Makefile (you can specify the target if needed)

# Fixes common InfiniBand errors
export UCX_TLS=tcp
export UCX_LOG_LEVEL=debug


# Step 3: Run Python script
echo "Running Python script..."
mpirun python3 pyTalbot/main.py