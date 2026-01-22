#!/bin/bash

START=${1}
END=${2}
DATASET=${3}  # "EVOCEANIA" or "1000g_gbr" etc.

for CHROM in {1..22}
do
    INDFILE=${DATASET}_chr${CHROM}_${START}_${END}
    if [ ! -e $INDFILE ]
    then
        echo "Re-making individual file $INDFILE"
        if [ $CHROM -ne 22 ]
        # if [ $CHROM -ne 21 ]
        then
            for pp in {250..299}
            do
                # echo results/realdata/${DATASET}/singer/observation_data/chr${CHROM}_t10000_strict_ind${START}_${END}.${pp}.npz >> $INDFILE
                echo results/realdata/${DATASET}/GhostHMM/chr${CHROM}/chr${CHROM}_t10000_strict_ind${START}_${END}.${pp}.npz >> $INDFILE
            done
        else
            # for pp in {224..273}
            # for pp in {237..286}
            for pp in {250..299}
            do
                # echo results/realdata/${DATASET}/singer/observation_data/chr${CHROM}_t10000_strict_ind${START}_${END}.${pp}.npz >> $INDFILE
                echo results/realdata/${DATASET}/GhostHMM/chr${CHROM}/chr${CHROM}_t10000_strict_ind${START}_${END}.${pp}.npz >> $INDFILE
            done
        fi
    fi
done

for (( k=START; k<=END; k++ ))
do
    indfile_string=$(printf "${DATASET}_chr%d_${START}_${END}," {1..22} | sed 's/,$//')
    gmap_string=$(printf "results/realdata/1000g_hg38_2022/genetic_map_hg38/genetic_map_hg38_chr%d.txt," {1..22} | sed 's/,$//')
    chrom_string=$(printf "chr%d," {1..22} | sed 's/,$//')
    out_string=$(printf "results/realdata/${DATASET}/GhostHMM/chr%d/singerave_t10000_full_ind," {1..22} | sed 's/,$//')
    python run_arghmm.py --data-file $indfile_string --individual ${k} --genetic-map $gmap_string --chrom $chrom_string --t-archaic 10000 --include-regions results/realdata/hg38_strictmask.bed --outpref $out_string &
done
wait
