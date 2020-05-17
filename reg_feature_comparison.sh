#!/bin/sh

arr1=(GB RF LR)
arr2=(20_80 50_50 80_20)

for i in 0 1 2
do
    for j in 0 1 2
    do
        for k in 1 2 3 4
        do 
            echo "classifier  ${arr1[i]}"
            echo "ratio:${arr2[j]} "
            echo "set: $k"
            python ml_reg.py -cls ${arr1[i]} -rat ${arr2[j]} -set $k

        done
    done
done