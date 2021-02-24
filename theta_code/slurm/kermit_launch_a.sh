#!/bin/bash
# Basic while loop
counter=1
while [ $counter -le 24 ]
  do
    echo $counter
    python grid_run.py $counter 2&>1 1&>"out.txt"
    wait
    ((counter++))
  done
echo All done
