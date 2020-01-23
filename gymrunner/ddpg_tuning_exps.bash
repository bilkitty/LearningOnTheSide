#!/bin/bash

names=(WindyGrid TaxiGrid CartPole Acrobot MountainCar Pendulumn MountainCarContinuous TheEnd)
algos=(Qlearning DDPG)

# Run mix of discrete and continuous environements
# TEMP: don't run discrete envs
algoType=1
resultsdir="../data/results/algo_tuning"

for param in h256b128d9t0 h16b128d5t0 h16b128d9t0 h16b20d5t0 h16b20d9t0 h256b128d5t0 h256b20d5t0 h256b20d9t h64b128d5t0 h64b128d9t0 h64b20d5t0 h64b20d9t0
do
  echo "Processing $param"
  trialdir="$resultsdir/$param"
  paramfile="../data/params/$param.json"
  mkdir -p $trialdir
  cp $paramfile "params.json"
  for env in 5 6
  do
    echo "Running ${algos[$algoType]} in ${names[$env]}"
    python3 main.py --envIndex $env --algoIndex $algoType > log.txt
    if [ $? -eq 0 ]
    then
      mv *.png $trialdir
      mv *.pkl $trialdir
      mv log.txt $trialdir/${algos[$algoType]}_${names[$env]}_log.txt
      cp "params.json" $trialdir
    else
      echo "Failed with exit code: $?"
    fi
    echo "==================================="
  done
done

echo "Done =^.^="
