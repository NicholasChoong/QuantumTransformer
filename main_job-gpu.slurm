#!/bin/bash -l
#SBATCH --job-name=amazon-amplitude
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=0
#SBATCH --gres=gpu:v100:1
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=21980614@student.uwa.edu.au
#SBATCH --time=3-00:00:00  

start_time=$(date)

# To configure GNU Environment for Mothur
module load Anaconda3/2024.06 cuda/12.4
eval "$(conda shell.bash hook)"

# activate the Python environment for the unit CITS5508
conda activate $MYGROUP/qt

# list the environment loaded by the modules.
# Can remove the two lines below if you want.
# module list
# conda list

# Note: SLURM_JOBID is a unique number for every job.
# These are generic variables.

# Below is the Python file that would be run. Replace
# lab05-sample.ipynb by your own file name.
# NAME=3_angle_rzx_basic_qvc
NAME=amazon-amplitude
EXT=ipynb
SCRIPT=$NAME.$EXT

SCRATCH=$MYSCRATCH/run_conda/$SLURM_JOBID
RESULTS=$MYGROUP/QuantumTransformer/conda_results

###############################################
# Creates a unique directory in the SCRATCH directory for this job to run in.
if [ ! -d $SCRATCH ]; then
  mkdir -p $SCRATCH
fi
echo Working SCRATCH directory is $SCRATCH

###############################################
# Creates a unique directory in your GROUP directory for the results of this job
if [ ! -d $RESULTS ]; then
  mkdir -p $RESULTS
fi
echo Results will be stored in $RESULTS/$SLURM_JOBID

#############################################
# Rather than copying the large CIFAR-10 batch files to the $SCRATCH directory
# (where the Python file will be run), we create symbolic links to the data files
# in that directory.

cd ${SLURM_SUBMIT_DIR}
echo "SLURM_SUBMIT_DIR is"
echo ${SLURM_SUBMIT_DIR}

# copy the mothur analysis script to SCRATCH

cp notebooks/${SCRIPT} ${SCRATCH}
# cp -r data/ ${SCRATCH}
cp -r transformer/ ${SCRATCH}
cp config.py ${SCRATCH}

# go to the /scratch... directory and create symbolic links to the
# files for the CIFAR-10 dataset and a link to Du's data_loader.py.
cd ${SCRATCH}

# we can delete the line below. It just shows the contents of
# the /scratch... directory before running Python.
ls -al

nvidia-smi

# now run our Python notebook file
papermill --request-save-on-cell-execute --progress-bar ./${SCRIPT} ./$NAME.papermill.$EXT

#############################################
# Now move the output produced by the Python notebook file from
# the /scratch... directory to my home directory.
rm -rf ${SCRATCH}/transformer

cd $HOME
mv ${SCRATCH} ${RESULTS}

echo "mv ${SCRATCH} ${RESULTS}"
echo "Please see the ${RESULTS} directory for any output"

echo
echo "Mothur MPI job started  at $start_time"
echo "Mothur MPI job finished at $(date)"
