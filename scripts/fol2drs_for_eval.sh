#!/bin/bash

outdir=$1

python deformation_tsv.py --input ${outdir} --format clf_eval
python DRS_parsing/counter.py -f1 ${outdir}/pred.clf -f2 ${outdir}/gold.clf  -ill dummy > ${outdir}/score.txt
