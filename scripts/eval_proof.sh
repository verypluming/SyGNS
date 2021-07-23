#!/bin/bash

vampire_dir=`cat vampire_location.txt`
txtfile=$1
type=$2
outdir=$3
mkdir -p $outdir


function mapping_to_fol(){
  formula=$1
  swipl -s fol2tptp.pl -g "fol2tptp(${formula},user)" -t halt --quiet
}

function str_to_list(){
 str=$1
 cat $str \
   | sed 's/ /,/g' \
   | sed -e 's/^/[/g' \
   | sed -e 's/$/]/g'
}

function call_vampire(){
  tptp=$1
  ${vampire_dir}/vampire $tptp \
    | head -n 1 \
    | awk '{if($0 ~ "Refutation found"){ans="entailment"} else {ans="neutral"} print ans}'
  }

IFS=$'\n'

id=0
corrab=0
corrba=0
corr=0
err=0
echo -e "id\tsentence\tgold=>pred\tpred=>gold\tphenomena_tags" > ${outdir}/results.txt
while read line; do \
  if [ "${id}" == "0" ]; then
      let id++
      continue
  fi
  sentid=$(echo $line | awk -F'\t' '{print $1}')
  sent=$(echo $line | awk -F'\t' '{print $3}')
  fol1=$(echo $line | awk -F'\t' '{print $4}')
  fol2=$(echo $line | awk -F'\t' '{print $5}')
  tag=$(echo $line | awk -F'\t' '{print $6}')
  sem1=${fol1}
  sem2=${fol2}
  if [ ${type} == "prolog" ]; then
    sem1=$(mapping_to_fol $fol1)
    sem2=$(mapping_to_fol $fol2)
  fi
 
  echo -e "fof(t,axiom, ${sem1})." \
    >> ${outdir}/${id}.tptp
  echo -e "fof(h,conjecture, ${sem2})." \
    >> ${outdir}/${id}.tptp

  answer1=$(call_vampire ${outdir}/${id}.tptp)

  echo -e "fof(t,axiom, ${sem2})." \
    >> ${outdir}/${id}_rev.tptp
  echo -e "fof(h,conjecture, ${sem1})." \
    >> ${outdir}/${id}_rev.tptp

  answer2=$(call_vampire ${outdir}/${id}_rev.tptp)

  if [ -z "${sem1}" ]; then
     let err++
     answer1="gold_error"
     answer2="gold_error"
  elif [ -z "${sem2}" ]; then
     let err++
     answer1="pred_error"
     answer2="pred_error"
  elif [ ${answer1} == "entailment" -a ${answer2} == "entailment" ]; then
     let corr++
  elif [ ${answer1} == "entailment" ]; then
     let corrab++
  elif [ ${answer2} == "entailment" ]; then
     let corrba++
  fi
  echo -e "${sentid}\t${sent}\t${answer1}\t${answer2}\t${tag}" >> ${outdir}/results.txt
  let id++
done < $txtfile

acc=$((corr*100/id))
accab=$((corrab*100/id))
accba=$((corrba*100/id))
err_prop=$((err*100/id))
echo "bidirectional acc: ${acc}% (${corr}/${id}), gold=>pred acc: ${accab}% (${corrab}/${id}), pred=>gold acc: ${accba}% (${corrba}/${id}), error cases: ${err_prop}% (${err}/${id})"
