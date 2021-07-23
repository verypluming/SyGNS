#!/bin/bash

# A script to generate sentences for semantic parsing experiments
#
# Usage:
# ./run_sp.sh <number of sentences> <directory name>

# Set the number of sentences generated for each depth
NUM=$1

# Set the directory name
DIR=$2
RES_DIR=${DIR}_results

mkdir -p $RES_DIR

# Generate sentence schema
./generate_sp.sh $DIR

# Instantiate by words in vocab.yaml
python instantiate_sp.py $DIR $NUM

# Parse each sentence to FOL and VF with phenomena tags
./get_semantics_sp.sh $RES_DIR

# Convert Prolog format to NLTK format and add DRS outputs
python deformation_sp.py ${RES_DIR}/results.tmp.tsv
