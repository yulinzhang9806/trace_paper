#!/bin/bash
#Job name:
#SBATCH --job-name=e_wafr
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
#SBATCH --ntasks=40
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

# sh extract_singer_real2.sh 200 249 current_batch_inds_249.txt ${1}
# sh extract_singer_real2.sh 250 299 current_batch_inds_299.txt ${1}
# sh extract_singer_real2.sh 300 349 current_batch_inds_349.txt ${1}
# sh extract_singer_real2.sh 350 397 current_batch_inds_397.txt ${1}

# echo "Starting job on ${1}" 
# sh extract_singer_real.sh 0 49 current_batch_inds_49.txt ${1}
# sh extract_singer_real.sh 50 99 current_batch_inds_99.txt ${1}
# sh extract_singer_real.sh 100 149 current_batch_inds_149.txt ${1}
# sh extract_singer_real.sh 150 197 current_batch_inds_197.txt ${1}


# sh extract_singer_real.sh 200 249 current_batch_inds_249.txt ${1}
# sh extract_singer_real.sh 250 299 current_batch_inds_299.txt ${1}
# sh extract_singer_real.sh 300 349 current_batch_inds_349.txt ${1}
# sh extract_singer_real.sh 350 399 current_batch_inds_399.txt ${1}

# sh extract_singer_real3.sh 0 49 current_batch_inds_49.txt ${1}
# sh extract_singer_real3.sh 50 99 current_batch_inds_99.txt ${1}
# sh extract_singer_real3.sh 100 149 current_batch_inds_149.txt ${1}
# sh extract_singer_real3.sh 100 149 current_batch_inds_149.txt ${1}
# sh extract_singer_real3.sh 150 213 current_batch_inds_213.txt ${1}

# sh extract_singer.sh 0 49 current_batch_inds_49.txt ${1}
# sh extract_singer.sh 50 99 current_batch_inds_99.txt ${1}
# for i in {1..10}
# do
#     sh runhmm.sh 0 49 chr1_full_files_49 $i
#     sh runhmm.sh 50 99 chr1_full_files_99 $i
# done
# wait

# sh run_hmm_real2.sh 200 249 EVOCEANIA
# sh run_hmm_real2.sh 250 299 EVOCEANIA
# sh run_hmm_real2.sh 300 349 EVOCEANIA
# sh run_hmm_real2.sh 350 397 EVOCEANIA

# sh run_hmm_real.sh 0 49 1000g_hg38_2022
# sh run_hmm_real.sh 50 99 1000g_hg38_2022
# sh run_hmm_real.sh 100 149 1000g_hg38_2022
# sh run_hmm_real.sh 150 197 1000g_hg38_2022

# sh run_hmm_real.sh 150 199 1000g_hg38_2022
# sh run_hmm_real.sh 200 249 1000g_hg38_2022
# sh run_hmm_real.sh 250 299 1000g_hg38_2022
# sh run_hmm_real.sh 300 349 1000g_hg38_2022
# sh run_hmm_real.sh 350 399 1000g_hg38_2022

# sh run_hmm_real3.sh 0 49 HGDP
# sh run_hmm_real3.sh 50 99 HGDP
# sh run_hmm_real3.sh 100 149 HGDP
# sh run_hmm_real3.sh 150 213 HGDP


# for pp in {224..273}
for pp in {250..299}
do
 	# python getdata_singer.py --tree-file results/realdata/1000g_hg38_2022/singer/LWK/lwk_chr${1}_${pp}.tsz --t-archaic 15000 --chrom chr${1} --outpref results/realdata/1000g_hg38_2022/singer/LWK/mutage_files/mutage_chr${1}_${pp} --individuals 0 --mutation-age &
    # python getdata_singer.py --tree-file results/realdata/EVOCEANIA/singer/EVO_YRI_full_chr${1}_${pp}.tsz --t-archaic 15000 --chrom chr${1} --outpref results/realdata/EVOCEANIA/singer/mutage_files/mutage_chr${1}_${pp} --individuals 0 --mutation-age &
    # wait
#    for i in {1..10}
#    do
#        python getdata_singer.py --tree-file results/realdata/test_simulation/lwk_ghost_model/singer/n100_seed${i}_${pp}.tsz --t-archaic 15000 --chrom 1 --outpref results/realdata/test_simulation/lwk_ghost_model/singer/n100_seed${i}_mutage_${pp} --individuals 0 --mutation-age &
#    done
    python getdata_singer.py --tree-file results/realdata/test_simulation/lwk_ghost_model/singer_afr/n100_seed${1}_${pp}.tsz --t-archaic 15000 --chrom 1 --outpref results/realdata/test_simulation/lwk_ghost_model/singer_afr/n100_seed${1}_mutage_${pp} --individuals 0 --mutation-age &
done
wait

