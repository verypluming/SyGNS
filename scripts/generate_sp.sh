#!/bin/bash

# Generate a set of schematic sentences
#
# Usage:
#
# Use cfg_sp.pl (CFG for semantic parsing)
# ./generate_sp.sh <dir_name>
#
# All results are to be stored in the <dir_name> directory

dir=$1

mkdir -p $dir

echo "Processing depth0..."
time swipl -s cfg_sp.pl -g "plain(0,1)" -t halt --quiet > ${dir}/depth0.scheme.txt
echo "Processing depth1..."
time swipl -s cfg_sp.pl -g "plain(1,1)" -t halt --quiet > ${dir}/depth1.scheme.txt
echo "Processing depth2..."
time swipl -s cfg_sp.pl -g "plain(2,1)" -t halt --quiet > ${dir}/depth2.scheme.txt
echo "Processing depth3..."
time swipl -s cfg_sp.pl -g "plain(3,2)" -t halt --quiet > ${dir}/depth3.scheme.txt
echo "Processing depth4..."
time swipl -s cfg_sp.pl -g "plain(4,3)" -t halt --quiet > ${dir}/depth4.scheme.txt

wc -l ${dir}/depth0.scheme.txt
wc -l ${dir}/depth1.scheme.txt
wc -l ${dir}/depth2.scheme.txt
wc -l ${dir}/depth3.scheme.txt
wc -l ${dir}/depth4.scheme.txt
