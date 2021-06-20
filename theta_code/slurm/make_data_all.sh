#!/bin/bash
# Basic while loop
python make_data.py 2&>1 1&>"d_out_1.txt" &
echo "launch 1"
sleep 20
python make_data_2.py 2&>1 1&>"d_out_2.txt" &
echo "launch 2"
sleep 20
python make_data_3.py 2&>1 1&>"d_out_3.txt" &
echo "launch 3"