
#salloc --time=5:00:0 --ntasks=4 --mem-per-cpu=125G --account=rpp-blairt2k --x11

module load singularity/3.6
module load intel/2018.3
module load openmpi/3.1.2
module load fftw-mpi/3.3.8
module load python/3.6.3
module load scipy-stack
module load mpi4py/3.0.3
module load hdf5-mpi/1.10.3

SCRIPT_TO_RUN=/bin/bash
if [[ $1 != "" ]]; then
	        SCRIPT_TO_RUN=$1
	fi

CONTAINER_PATH=/project/6008045/machine_learning/containers/base_ml_recommended.simg
SCRATCH_DIR=`readlink -f /scratch/${USER}`
PROJECT_DIR=`readlink -f /project/6008045`
DATA_DIR=`readlink -f /scratch/prouse`

singularity exec --nv --bind ${PROJECT_DIR} --bind ${SCRATCH_DIR} --bind ${DATA_DIR} ${CONTAINER_PATH} ${SCRIPT_TO_RUN}
