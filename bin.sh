#!/bin/sh


# python ml_bin.py -cls DNN -set 1 -file 1
arr=(DNN GB RF LR)

for h in 1 2 3 4 5
do
    for i in 0 1 2 3
    do
       for j in 1 2 3 4
       do
           echo "setting number is $j"
           echo "classifier: ${arr[i]}"
           python ml_bin.py -cls ${arr[i]} -set $j -file $h

       done
    done
done
