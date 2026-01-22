#!/bin/bash

START=${1}
END=${2}
INDFILE=${3}
CHROM=${4}


rm $INDFILE
for pp in {250..299}
# for pp in {224..273}
do
#    echo results/realdata/EVOCEANIA/GhostHMM/chr${CHROM}/chr${CHROM}_t15000_strict_ind${START}_${END}.${pp}.npz >> $INDFILE
    echo results/realdata/test_simulation/lwk_ghost_model/singer_afr/n100_seed${CHROM}_t15000_ind${START}_${END}.${pp}.npz >> $INDFILE
done

for (( k=START; k<=END; k++ ))
do
    # python run_arghmm.py --data-file $INDFILE --individual ${k} --chrom chr1 --t-archaic 10000 --outpref results/realdata/test_simulation/lwk_ghost_model/singer/n100_seed${CHROM}_t10000_ind &
    python run_arghmm.py --data-file $INDFILE --individual ${k} --chrom chr1 --t-archaic 15000 --outpref results/realdata/test_simulation/lwk_ghost_model/singer_afr/n100_seed${CHROM}_t15000_ind$((k+200)) &
    # python run_arghmm.py --data-file $INDFILE --individual ${k} --chrom chr1 --t-archaic 15000 --outpref results/realdata/test_simulation/eur_ghost_recrate_model/singer/n100_seed${CHROM}_t15000_constant_ind & #--genetic-map results/realdata/1000g_hg38_2022/genetic_map_hg38/genetic_map_hg38_chr21.txt
done
wait
