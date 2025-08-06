#!/bin/bash
#SBATCH -A pccr
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -p cpu
#SBATCH -q normal
#SBATCH -t 24:00:00
#SBATCH --mail-user=verma198@purdue.edu
#SBATCH --mail-type=FAIL

starts=$(date +"%s")
start=$(date +"%r, %m-%d-%Y")

module load conda
conda activate /depot/natallah/data/shourya/flowmomo_env

# Change to your project directory
cd /depot/natallah/data/shourya/flowmomo

python -m scripts.metrics.peptide --results ./results/pep/results.jsonl --num_workers 128
# python -m scripts.metrics.peptide --results ./results/ab/results.jsonl --antibody --log_suffix HCDR3 --num_workers 128

ends=$(date +"%s")
end=$(date +"%r, %m-%d-%Y")
diff=$(($ends-$starts))
hours=$(($diff / 3600))
dif=$(($diff % 3600))
minutes=$(($dif / 60))
seconds=$(($dif % 60))

printf "\n\t===========Time Stamp===========\n"
printf "\tStart\t:$start\n\tEnd\t:$end\n\tTime\t:%02d:%02d:%02d\n" "$hours" "$minutes" "$seconds"
printf "\t================================\n\n"

sacct --jobs=$SLURM_JOBID --format=jobid,jobname,qos,nnodes,ncpu,maxrss,cputime,avecpu,elapsed