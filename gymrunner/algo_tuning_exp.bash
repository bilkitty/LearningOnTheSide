#!/bin/bash

names=(WindyGrid TaxiGrid CartPole Acrobot MountainCar ContinuousPendulum MountainCarContinuous TheEnd)
algos=(Qlearning ddpg)

# Run mix of discrete and continuous environements
# TEMP: don't run discrete envs
algoType=1
resultsdir="../data/results/DDPG/postrefactor"
configdir="config"

# Try low explore rate
for config in h256b128d99s01 h256b128d99s01 h256b128d99s30 h64b128d99s30
do
  echo "Processing $config"
  trialdir="$resultsdir"
  mkdir -p $trialdir
  for env in 6
  do
    trialfilename=$trialdir/${algos[$algoType]}_${names[$env]}_$config
    cp "$configdir/$config.json" "params.json"
    echo "Running ${algos[$algoType]} in ${names[$env]}"
    python3 main.py --envIndex $env > "$trialfilename.txt"
    if [ $? -eq 0 ]
    then
      mv *.png $trialdir
      mv *.pkl $trialdir
      cp "params.json" "$trialfilename.json"
    else
      echo "Failed with exit code: $?"
    fi
    echo "==================================="
  done
done


echo "Done =^.^="
