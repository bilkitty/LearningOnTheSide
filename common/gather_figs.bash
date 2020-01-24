#!/bin/bash

basePath=$1
dstPath=$2
pattern=$3 # should be last arg to accomodate empty pattern
tmpPath=$dstPath/tmpfigs
dirCount=0
echo "Searching for figs in:"
for item in $basePath/$pattern*
do
  if [ -d $item ]
  then
    echo "  '$item'"
    ((dirCount = dirCount + 1))

    mkdir -p $tmpPath
    cp $item/*.png $tmpPath

  fi
done

if [ $dirCount -eq 0 ]
then
  echo "No directories to search"
fi