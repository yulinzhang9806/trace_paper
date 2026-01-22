#!/bin/bash
#Job name:
#SBATCH --job-name=summary
#
# Account:
#SBATCH --account=fc_moorjani
#
# Partition:
#SBATCH --partition=savio2_bigmem
#
##QoS:
##SBATCH --qos=savio_normal
#
# Request one node:
#SBATCH --nodes=1
#
# Specify one task:
##SBATCH --ntasks=40
#
# Number of processors for single task needed for use case:
#SBATCH --cpus-per-task=1
#
# Wall clock limit:
#SBATCH --time=72:00:00
#
##SBATCH -C savio4_m512
##SBATCH --mem-per-cpu=8G
## Command(s) to run :
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export OPENBLAS_NUM_THREADS=1
module load bio/bcftools 
module load bio/vcftools
module load parallel

# python mut.py ${1}

summarypref="results/realdata/1000g_hg38_2022/GhostHMM/LWK/allchr/"
pop="lwk"
xsspref="results/realdata/1000g_hg38_2022/GhostHMM/LWK"
python append_snpinfo2.py ${summarypref}singerave_t15000_${pop}.new.count_merge.txt ${summarypref}singerave_t15000_${pop}.manifesto.count_merge.txt ${xsspref} ${pop} Ghost

# summarypref="results/realdata/1000g_hg38_2022/GhostHMM/LWK/chr${1}/"
# filepref="singerave_t15000_ind"
# # rm chr${1}_full_files_49
# # for pp in {250..299}
# # do
# #     echo results/realdata/1000g_hg38_2022/GhostHMM/singer_full_phase3samples/chr${1}/chr${1}_t15000_strict_ind0_49.${pp}.npz >> chr${1}_full_files_49
# # done
# # python run_arghmm.py --data-file chr${1}_full_files_49 --individual 0 --genetic-map results/realdata/1000g_hg38_2022/genetic_map_hg38/genetic_map_hg38_chr${1}.txt --chrom chr${1} --t-archaic 15000 --include-regions results/realdata/hg38_strictmask.bed --outpref results/realdata/1000g_hg38_2022/GhostHMM/singer_full_phase3samples/chr${1}/singerave_t15000_chr${1}_ind
# # python summarize.py --file ${summarypref}${filepref}0.chr${1}.xss.npz --chrom chr${1} --out ${summarypref}${filepref}0
# # python append_snpinfo.py 0 ${summarypref} ${filepref} chr${1}
# for (( i=0; i<400; i+=40 ))
# do
# 	for (( j=i; j<$(( i+40 )); j++ ))
# 	do
# 		# python summarize.py --file ${summarypref}${filepref}${j}.chr${1}.xss.npz --chrom chr${1} --out ${summarypref}${filepref}${j} &
# 		python append_snpinfo.py ${j} ${summarypref} ${filepref} chr${1} &
# 	done
# 	wait
# done
# wait

#summarypref="results/realdata/1000g_hg38_2022/GhostHMM/singer_full_phase3samples/"
#filepref="singerave_t50000_chr${1}_ind"
#for (( i=0; i<398; i+=40 ))
#do
#        python summarize.py --file ${summarypref}${filepref}${i}.chr${1}.xss.npz --out ${summarypref}${filepref}${i} --genetic-distance-threshold 0.01 --physical-length-threshold 15000
#	for (( j=i; j<$(( i+40 )); j++ ))
#	do
#		python summarize.py --file ${summarypref}${filepref}${j}.chr${1}.xss.npz --out ${summarypref}${filepref}${j} --genetic-distance-threshold 0.02 --physical-length-threshold 20000
#        	python append_snpinfo.py ${j} ${summarypref} ${filepref} &
#	done
#	wait
#done
#wait
 
