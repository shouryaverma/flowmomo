#!/bin/bash
#SBATCH -A pccr
#SBATCH -N 1
#SBATCH -p ai
#SBATCH -q preemptible
#SBATCH -t 4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=14
#SBATCH --mail-user=verma198@purdue.edu
#SBATCH --mail-type=FAIL

starts=$(date +"%s")
start=$(date +"%r, %m-%d-%Y")

module load conda
conda activate /depot/natallah/data/shourya/flowmomo_env
# source $HOME/.bashrc
export CUDA_LAUNCH_BLOCKING=1

# Change to your project directory
cd /depot/natallah/data/shourya/flowmomo

# # peptide
python generate.py --config configs/test/test_pep.yaml --ckpt /depot/natallah/data/shourya/flowmomo/ckpts/unimomo/Flow_unimomo/version_0/checkpoint/epoch39_step9480.ckpt --gpu 0 --save_dir ./results/pep
# # antibody
# python generate.py --config configs/test/test_ab.yaml --ckpt /depot/natallah/data/shourya/flowmomo/ckpts/unimomo/Flow_unimomo/previous1/checkpoint/epoch136_step32469.ckpt --gpu 0 --save_dir ./results/ab
# # small molecule
# python generate.py --config configs/test/test_mol.yaml --ckpt /depot/natallah/data/shourya/flowmomo/ckpts/unimomo/Flow_unimomo/previous1/checkpoint/epoch136_step32469.ckpt --gpu 0 --save_dir ./results/mol

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

# sinteractive -A pccr -N 1 -p ai -q preemptible -t 4:00:00 --gpus-per-node=1 --cpus-per-gpu=14 --mail-user=verma198@purdue.edu --mail-type=FAIL
# sinteractive -A pccr -N 1 -p cpu -q normal -t 2:00:00 --mail-user=verma198@purdue.edu --mail-type=FAIL