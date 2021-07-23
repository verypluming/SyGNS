#!/bin/bash
#
# Download Vampire (Version 4.4) from https://github.com/vprover/vampire
#

vampire_url="https://github.com/vprover/vampire/archive/4.4.tar.gz"
vampire_basename=`basename $vampire_url`

if [ ! -d vampire ]; then
  curl -LO $vampire_url
  tar -zxvf $vampire_basename
fi

# Set path to vampire-4.4 directory
vampire_dir=`pwd`"/"vampire-4.4
echo $vampire_dir > scripts/vampire_dir.txt

rm -f $vampire_basename

# Make release version
cd ${vampire_dir}
make vampire_rel
cp vampire_rel_* vampire
echo `pwd`"/"vampire-4.4 > vampire_location.txt

pip install pandas nltk yaml
