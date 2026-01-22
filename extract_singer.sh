#!/bin/bash

START=${1}
END=${2}
INDFILE=${3}
SEED=${4}

if [ `wc -l $INDFILE | awk '{print $1}'` -ne $(( END - START + 1 )) ] || [ `head -1 $INDFILE` -ne $START ] || [ `tail -1 $INDFILE` -ne $END ]
then
    echo "Re-making individual file $INDFILE"
    rm $INDFILE
    for (( j=START; j<=END; j++ ))
    do
        echo $j >> $INDFILE
    done
fi

for (( k=250; k<300; k+=25 ))
do
    for (( pp=$k; pp<$(( k + 25 )); pp+=1 ))
    do
        # python getdata_singer.py --tree-file results/realdata/test_simulation/eur_ghost_human_model/singer/n100_seed${SEED}_${pp}.tsz --t-archaic 15000 --chrom chr1 --outpref results/realdata/test_simulation/eur_ghost_human_model/singer/n100_seed${SEED}_t15000_ind${START}_${END}.${pp} --individual-file $INDFILE &
        python getdata_singer.py --tree-file results/realdata/test_simulation/lwk_ghost_model/singer_afr/n100_seed${SEED}_${pp}.tsz --t-archaic 15000 --chrom chr1 --outpref results/realdata/test_simulation/lwk_ghost_model/singer_afr/n100_seed${SEED}_t15000_ind${START}_${END}.${pp} --individual-file $INDFILE &
    done
    wait
done
wait	
