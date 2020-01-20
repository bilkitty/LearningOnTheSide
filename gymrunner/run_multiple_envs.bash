#!/bin/bash

names=(WindyGrid TaxiGrid CartPole Acrobot MountainCar MountainCarContinuous Pendulumn TheEnd)
algos=(Qlearning, DDPG)

# Run simple environments
algoType=0
resultsdir="../data/results/QLearning"
mkdir -p $resultsdir
for env in 0 1 2
do
  echo "Running ${algos[$algoType]} in ${names[$env]}"
  python3 main.py $env > log.txt
  if [ $? -eq 0 ]
  then
    mv *.png $resultsdir
    mv *.pkl $resultsdir
    mv log.txt $resultsdir/${algos[$algoType]}_${names[$env]}Env_log.txt
  else
    echo "Failed with exit code: $?"
  fi
  echo "==================================="
done

# Run harder environements
algoType=0
resultsdir="../data/results/QLearning"
mkdir -p $resultsdir
for env in 3 4
do
  echo "Running ${algos[$algoType]} in ${names[$env]}"
  python3 main.py $env 5000 > log.txt
  if [ $? -eq 0 ]
  then
    mv *.png $resultsdir
    mv *.pkl $resultsdir
    mv log.txt $resultsdir/${algos[$algoType]}_${names[$env]}Env_log.txt
  else
    echo "Failed with exit code: $?"
  fi
  echo "==================================="
done

# Run mix of discrete and continuous environements
# TEMP: don't run discrete envs
algoType=1
resultsdir="../data/results/DDPG"
mkdir -p $resultsdir
for env in 5 6
do
  echo "Running ${algos[$algoType]} in ${names[$env]}"
  python3 main.py $env 1 5000 > log.txt
  if [ $? -eq 0 ]
  then
    mv *.png $resultsdir
    mv *.pkl $resultsdir
    mv log.txt $resultsdir/${algos[$algoType]}_${names[$env]}Env_log.txt
  else
    echo "Failed with exit code: $?"
  fi
  echo "==================================="
done


echo "Done =^.^="
