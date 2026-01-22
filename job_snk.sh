#!/bin/bash
#Job name:
#SBATCH --job-name=snk
#
# Account:
#SBATCH --account=co_moorjani
#
# Partition:
#SBATCH --partition=savio4_htc
#
##QoS:
##SBATCH --qos=savio_normal
#
# Request one node:
#SBATCH --nodes=1
#
# Specify one task:
#SBATCH --ntasks=50
#
# Number of processors for single task needed for use case:
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=72:00:00
#
#SBATCH -C savio4_m512
#SBATCH --mem-per-cpu=8G
## Command(s) to run :
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
module load bio/bcftools 
module load bio/vcftools
module load parallel
#LC_ALL=en_CA.UTF-8 snakemake -c 20 --rerun-incomplete
# snakemake --unlock
snakemake --retries -3 -c all --rerun-incomplete
#snakemake -R $(snakemake --list-code-changes) --touch -c 40 --rerun-incomplete
