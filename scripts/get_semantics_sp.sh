#!/bin/bash

res_dir=$1

mkdir -p $res_dir

function sent2fol(){
  depth=$1
  sentence=$2
  swipl -s cfg_sp_sem.pl -g "sent2fol(${depth},${sentence})" -t halt --quiet
}

function sent2phen(){
  depth=$1
  sentence=$2
  swipl -s cfg_sp_sem.pl -g "sent2phen(${depth},${sentence})" -t halt --quiet
}

function sent2vf(){
  depth=$1
  sentence=$2
  swipl -s cfg_sp_sem.pl -g "sent2vf(${depth},${sentence})" -t halt --quiet
}

function sent2vfpol(){
  depth=$1
  sentence=$2
  swipl -s cfg_sp_sem.pl -g "sent2vfpol(${depth},${sentence})" -t halt --quiet
}

function sent2folphen(){
  depth=$1
  sentence=$2
  swipl -s cfg_sp_sem.pl -g "sent2folphen(${depth},${sentence})" -t halt --quiet
}

function str_to_list(){
 str=$1
 cat $str \
   | sed 's/ /,/g' \
   | sed -e 's/^/[/g' \
   | sed -e 's/$/]/g'
}

IFS=$'\n'
id=0

echo -e "id\tdepth\tsentence\tfol\tvf\tvfpol\ttags" \
    > $res_dir/results.tmp.tsv

for txtfile in `ls ${res_dir}/depth*.txt`
do
  filename=${txtfile##*/}
  depth=$(echo "${filename:5:1}")
  while read line; do \
    let id++
    sent=$(echo ${line} | str_to_list)
    echo "Processing depth$depth-$id"
    folphen=$(sent2folphen ${depth} ${sent})
    vf=$(sent2vf ${depth} ${sent})
    vfpol=$(sent2vfpol ${depth} ${sent})
    fol=$(echo ${folphen} | awk -F'@' '{print $1}')
    phen=$(echo ${folphen} | awk -F'@' '{print $2}')

    echo -e "${id}\t${depth}\t${line}\t${fol}\t${vf}\t${vfpol}\t${phen}" \
      >> $res_dir/results.tmp.tsv
  done < $txtfile
done
