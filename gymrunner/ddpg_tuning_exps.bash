#!/bin/bash

names=(WindyGrid TaxiGrid CartPole Acrobot MountainCar Pendulumn MountainCarContinuous TheEnd)
algos=(Qlearning, DDPG)

# Run mix of discrete and continuous environements
# TEMP: don't run discrete envs
algoType=1
resultsdir="../data/results/algo_tuning"
mkdir -p $resultsdir
for env in 5
do
  echo "Running ${algos[$algoType]} in ${names[$env]}"
  python3 main.py $env $algoType 500 > log.txt
  if [ $? -eq 0 ]
  then
    mv *.png $resultsdir
    mv *.pkl $resultsdir
    mv log.txt $resultsdir/${algos[$algoType]}_${names[$env]}_log.txt
  else
    echo "Failed with exit code: $?"
  fi
  echo "==================================="
done


echo "Done =^.^="
