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
#SBATCH --ntasks=10
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

# python sim_msprime.py lwk_ghost ${1}

# for i in {250..274}
# do
#     python extract_tmrca.py results/realdata/1000g_hg38_2022/singer/polarized_full_phase3samples/yri_gbr_chr${1}_${i}.tsz results/tmrca/realdata/1000g/chr${1}_yri_tmrca_eur_ind283_397_${i} 283 398 &
# done
# wait
# for i in {275..299}
# do
#     python extract_tmrca.py results/realdata/1000g_hg38_2022/singer/polarized_full_phase3samples/yri_gbr_chr${1}_${i}.tsz results/tmrca/realdata/1000g/chr${1}_yri_tmrca_eur_ind283_397_${i} 283 398 &
# done
# wait

for i in {200..249}
do
    # for j in {1..10}
    # do
    #     python summarize.py --file results/realdata/test_simulation/lwk_ghost_model/singer_afr/n100_seed${j}_t15000_ind${i}.chr1.xss.npz --out results/realdata/test_simulation/lwk_ghost_model/singer_afr/n100_seed${j}_t15000_ind${i} --chrom 1 &
	#     # python sim.py $i $j SUPER lwk_ghost_model &
    #     # python sim_annote.py $i $j Ghost eur_ghost_model &
    # done
    python sim_annote.py $i ${1} Ghost lwk_ghost_model &
done
wait
for i in {250..299}
do
    # for j in {1..10}
    # do
    #     python summarize.py --file results/realdata/test_simulation/lwk_ghost_model/singer_afr/n100_seed${j}_t15000_ind${i}.chr1.xss.npz --out results/realdata/test_simulation/lwk_ghost_model/singer_afr/n100_seed${j}_t15000_ind${i} --chrom 1 &
	#     # python sim.py $i $j SUPER lwk_ghost_model &
    #     # python sim_annote.py $i $j Ghost eur_ghost_model &
    # done
    python sim_annote.py $i ${1} Ghost lwk_ghost_model &
done
wait
