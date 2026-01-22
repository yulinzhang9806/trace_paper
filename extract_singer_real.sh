#!/bin/bash

START=${1}
END=${2}
INDFILE=${3}
CHROM=${4}

if [ `wc -l $INDFILE | awk '{print $1}'` -ne $(( END - START + 1 )) ] || [ `head -1 $INDFILE` -ne $START ] || [ `tail -1 $INDFILE` -ne $END ]
then
    echo "Re-making individual file $INDFILE"
    rm $INDFILE
    for (( j=START; j<=END; j++ ))
    do
        echo $j >> $INDFILE
    done
fi

# pp=299
# python getdata_singer.py --tree-file results/realdata/EVOCEANIA/singer/EVO_YRI_chr${CHROM}_${pp}.tsz --t-archaic 10000 --chrom chr${CHROM} --include-regions results/realdata/hg38_strictmask.bed --outpref results/realdata/EVOCEANIA/GhostHMM/chr${CHROM}/chr${CHROM}_t10000_strict_ind${START}_${END}.${pp} --individual-file $INDFILE

# for (( k=224; k<274; k+=15 ))
for (( k=250; k<300; k+=25 ))
do
    if [ $(( k + 25 )) -gt 300 ]
    then
        kk=300
    else
        kk=$(( k + 25 ))
    fi
    for (( pp=$k; pp<$kk; pp+=1 ))
    do
	    python getdata_singer.py --tree-file results/realdata/1000g_hg38_2022/singer/LWK_tsz/lwk_chr${CHROM}_${pp}.tsz --t-archaic 15000 --chrom chr${CHROM} --include-regions results/realdata/hg38_strictmask.bed --outpref results/realdata/1000g_hg38_2022/GhostHMM/LWK/chr${CHROM}/chr${CHROM}_t15000_strict_ind${START}_${END}.${pp} --individual-file $INDFILE &
	    # python getdata_singer.py --tree-file results/realdata/EVOCEANIA/singer/EVO_YRI_chr${CHROM}_${pp}.tsz --t-archaic 10000 --chrom chr${CHROM} --include-regions results/realdata/hg38_strictmask.bed --outpref results/realdata/EVOCEANIA/GhostHMM/chr${CHROM}/chr${CHROM}_t10000_strict_ind${START}_${END}.${pp} --individual-file $INDFILE &
    done
    wait
done
wait	
