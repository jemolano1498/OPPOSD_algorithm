#!/bin/bash

for n in {1..5};
do
  sbatch run_script.sh opposd 5000 500 $n
  sbatch run_script.sh offpac 5000 500 $n
done


