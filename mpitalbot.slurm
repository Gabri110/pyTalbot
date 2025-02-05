#!/bin/bash
#SBATCH --job-name=talbot_job     # Job name
#SBATCH --output=talbot_job%j.txt # Standard output and error log (%j is the job ID)

#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --ntasks=4                # Number of tasks
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB                 # Total memory for the job

#SBATCH --time=100:00:00          # Maximum time for the job (HH:MM:SS)
#SBATCH --partition=general       # Specify the partition (modify if needed)
#SBATCH --account=gybarra         # Account name (modify if needed)

# Print information about the job and the node
echo "Starting SLURM job..."
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Step 1: Load necessary modules (modify based on system configuration)
module load Python      # Load the Anaconda module if it's available
module load gsl         # Load the GSL module (check availability)
module load ffmpeg      # Load the FFmpeg module (check availability)

# Step 2: Run Makefile (compiling code using make)
echo "Running Makefile..."
make -C pyTalbot/pyTalbot/src/  # Run the Makefile (you can specify the target if needed)

# Step 3: Run Python script
echo "Running Python script..."
mpirun python3 pyTalbot/main.py

echo "Job completed successfully!"
