#!/bin/bash
# Basic while loop
counter=0
while [ $counter -le 12 ]
  do
    echo $counter
    python grid_run.py $counter
    #python grid_run.py $counter 2&>1 1&>"out.txt"
    wait
    ((counter++))
  done
echo All done
