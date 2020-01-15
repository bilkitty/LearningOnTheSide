#!/bin/bash

names=(WindyGrid TaxiGrid CartPole Acrobot MountainCar TheEnd)
mkdir -p QLearningResults
# Run simple environments
for env in 0 1 2
do
  echo "Running qlearning in ${names[$env]}"
  python3 main.py $env > log.txt
  if [ $? -eq 0 ]
  then
    mv *.png QLearningResults
    mv *.pkl QLearningResults
    mv log.txt QLearningResults/${names[$env]}Env_log.txt
  else
    echo "Failed with exit code: $?"
  fi
  echo "==================================="
done

# Run harder environements
for env in 3 4
do
  echo "Running qlearning in ${names[$env]}"
  python3 main.py $env 5000 > log.txt
  if [ $? -eq 0 ]
  then
    mv *.png QLearningResults
    mv *.pkl QLearningResults
    mv log.txt QLearningResults/${names[$env]}Env_log.txt
  else
    echo "Failed with exit code: $?"
  fi
  echo "==================================="
done

echo "Done =^.^="
