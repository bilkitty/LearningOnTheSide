#!/bin/bash

names=(WindyGrid TaxiGrid CartPole Acrobot MountainCar Pendulumn MountainCarContinuous TheEnd)
algos=(Qlearning DDPG)

# Run mix of discrete and continuous environements
# TEMP: don't run discrete envs
algoType=1
resultsdir="../data/results/algo_tuning"

# Try low tau
# All configs: h256b128d9t0 h256b20d5t0 h256b128d5t0 h256b20d9t0 h64b128d5t0 h64b128d9t0 h64b20d5t0 h64b20d9t0 h16b128d5t0 h16b128d9t0 h16b20d5t0 h16b20d9t0
for param in h256b128d5t0 h256b20d9t0 h64b128d5t0 h64b128d9t0 h16b20d5t0 h16b20d9t0
do
  echo "Processing $param"
  tau=0.01
  tauVal="tau01"
  trialdir="$resultsdir/$param/$tauVal"
  paramfile="../data/params/$param.json"
  mkdir -p $trialdir
  cp $paramfile "params.json"
  for env in 6
  do
    echo "Running ${algos[$algoType]} in ${names[$env]}"
    python3 main.py --envIndex $env --algoIndex $algoType --tau $tau > log.txt
    if [ $? -eq 0 ]
    then
      mv *.png $trialdir
      mv *.pkl $trialdir
      mv log.txt $trialdir/${algos[$algoType]}_${names[$env]}_log.txt
      cp "params.json" "$trialdir/params_$tauVal.json"
    else
      echo "Failed with exit code: $?"
    fi
    echo "==================================="
  done
done

# Try high tau
# All configs: h256b128d9t0 h256b20d5t0 h256b128d5t0 h256b20d9t0 h64b128d5t0 h64b128d9t0 h64b20d5t0 h64b20d9t0 h16b128d5t0 h16b128d9t0 h16b20d5t0 h16b20d9t0
for param in h256b128d5t0 h256b20d9t0 h64b128d5t0 h64b128d9t0 h16b20d5t0 h16b20d9t0
do
  echo "Processing $param"
  tau=0.1
  tauVal="tau10"
  trialdir="$resultsdir/$param/$tauVal"
  paramfile="../data/params/$param.json"
  mkdir -p $trialdir
  cp $paramfile "params.json"
  for env in 6
  do
    echo "Running ${algos[$algoType]} in ${names[$env]}"
    python3 main.py --envIndex $env --algoIndex $algoType --tau $tau > log.txt
    if [ $? -eq 0 ]
    then
      mv *.png $trialdir
      mv *.pkl $trialdir
      mv log.txt $trialdir/${algos[$algoType]}_${names[$env]}_log.txt
      cp "params.json" "$trialdir/params_$tauVal.json"
    else
      echo "Failed with exit code: $?"
    fi
    echo "==================================="
  done
done


echo "Done =^.^="
