#!/bin/bash

names=(WindyGrid TaxiGrid CartPole Acrobot MountainCar ContinuousPendulum MountainCarContinuous TheEnd)
algos=(Qlearning ddpg)

# Run mix of discrete and continuous environements
# TEMP: don't run discrete envs
algoType=1
resultsdir="../data/results/algo_tuning/working"
configdir="config"

# Try low explore rate
for config in ddpg_ace_pendulum_h256b128d99 h64b128d99s01 h256b128d99s30 h256b128d50s01
do
  echo "Processing $config"
  trialdir="$resultsdir"
  mkdir -p $trialdir
  for env in 5 6
  do
    cp "$configdir/$config.json" "params.json"
    echo "Running ${algos[$algoType]} in ${names[$env]}"
    python3 main.py --envIndex $env --algoIndex $algoType > "$trialdir/$config.txt"
    if [ $? -eq 0 ]
    then
      mv *.png $trialdir
      mv *.pkl $trialdir
      cp "params.json" "$trialdir/$config.json"
    else
      echo "Failed with exit code: $?"
    fi
    echo "==================================="
  done
done


echo "Done =^.^="
