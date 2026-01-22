#!/bin/bash

chromo=${1}
popname=${2}
sub=${3}
argweaver_path="/global/home/users/zhangyulin9806/argweaver-master/bin"
input_sites="argweaver/chr${chromo}_${popname}_sub.${sub}.sites"
outpref="argweaver/ppoutput/chr${chromo}_${popname}_sub.${sub}"
Ne=10000
mu=1.25e-8
rec_rate=1e-8
ntimes=20
maxtime=200e4
compress=10
n_iter=2000
seed=${4}

${argweaver_path}/arg-sample -s $input_sites -N $Ne -r $rec_rate -m $mu --ntimes $ntimes --maxtime $maxtime -c $compress -n $n_iter --randseed $seed -o $outpref
