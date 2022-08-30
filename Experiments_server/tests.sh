#!/bin/bash

# OFFPAC Random batch
#sbatch run_script.sh offpac random 100000 500 5000
#sbatch run_script.sh opposd random 100000 500 5000
#sbatch run_script.sh offpac random 2500 500
#sbatch run_script.sh offpac random 5000 500
#sbatch run_script.sh offpac random 10000 500
#sbatch run_script.sh offpac random 25000 500
#sbatch run_script.sh offpac random 50000 500
#sbatch run_script.sh offpac random 100000 500 #85
## OPPOSD Random batch
#sbatch run_script.sh opposd random 1000 500 #86
#sbatch run_script.sh opposd random 2500 500
#sbatch run_script.sh opposd random 5000 500
#sbatch run_script.sh opposd random 10000 500
#sbatch run_script.sh opposd random 25000 500
#sbatch run_script.sh opposd random 50000 500
#sbatch run_script.sh opposd random 100000 500 #92
#
## OFFPAC experiment batch
#sbatch run_script.sh offpac experiment 2000 500 #93 <--
#sbatch run_script.sh offpac experiment 3000 500
#sbatch run_script.sh offpac experiment 6000 500
#sbatch run_script.sh offpac experiment 12000 500
#sbatch run_script.sh offpac experiment 24000 500 #97
## OPPOSD experiment batch

#for n in {1..5};
#do
#  sbatch run_script.sh opposd experiment 2000 500 $n
#  sbatch run_script.sh opposd experiment 3000 500 $n
#  sbatch run_script.sh opposd experiment 6000 500 $n
#  sbatch run_script.sh opposd experiment 12000 500 $n
#  sbatch run_script.sh opposd experiment 24000 500 $n
#done

#for n in {1..5};
#do
#  sbatch run_script.sh offpac random 100000 500 $n 5000
#  sbatch run_script.sh opposd random 100000 500 $n 5000
#done

#for n in {1..5};
#do
#  sbatch run_script.sh offpac experiment 2000 500 $n
#  sbatch run_script.sh offpac experiment 3000 500 $n
#  sbatch run_script.sh offpac experiment 6000 500 $n
#  sbatch run_script.sh offpac experiment 12000 500 $n
#  sbatch run_script.sh offpac experiment 24000 500 $n
#done
#
#for n in {1..5};
#do
#  sbatch run_script.sh opposd random 1000 500 $n
#  sbatch run_script.sh opposd random 2500 500 $n
#  sbatch run_script.sh opposd random 5000 500 $n
#  sbatch run_script.sh opposd random 10000 500 $n
#  sbatch run_script.sh opposd random 25000 500 $n
#  sbatch run_script.sh opposd random 50000 500 $n
#  sbatch run_script.sh opposd random 100000 500 $n
#done

for n in {1..10};
do
  sbatch run_script.sh opposd random 2500 1000 $n
  sbatch run_script.sh offpac random 2500 1000 $n
  sbatch run_script.sh opposd experiment 2500 1000 $n
  sbatch run_script.sh offpac experiment 2500 1000 $n
done





