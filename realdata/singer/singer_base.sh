#!/bin/bash

#SBATCH --job-name=CHROM_JOBNAME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=54:00:00
#SBATCH --mem=10G
#SBATCH -o slurm-JOBNAME.%j.out

## Command(s) to run :
singer_master=/global/scratch/users/zhangyulin9806/github/singer-0.1.8-beta-linux-x86_64/singer_master
vcf=/path/to/input/folder/1000g_hg38_chrCHROM_polarized
outpref=/path/to/output/folder/yri_chrCHROM_JOBNAME
start=START
end=END
mu=1.25e-8
nsamp=300
thin=100
ne=2e4
${singer_master} -vcf ${vcf} -m ${mu} -start ${start} -end ${end} -Ne ${ne} -n ${nsamp} -thin ${thin} -polar 0.99 -output ${outpref}

