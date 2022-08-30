#!/bin/bash

for n in {1..5};
do
  sbatch run_script.sh RL 1000 $n
  sbatch run_script.sh AC 1000 $n
  sbatch run_script.sh OFFPAC 1000 $n
  sbatch run_script.sh PPO 1000 $n
done
