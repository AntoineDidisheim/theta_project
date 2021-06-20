#!/bin/bash
# Basic while loop
counter=0
while [ $counter -le 26 ]
  do
    echo $counter
    python make_hist_theta.py $counter 2&>1 1&>"out_$counter.txt"
    disown -a
    ((counter++))
    sleep 10
  done
echo All done
