#!/bin/sh

arr1=(GB RF LR)
arr2=(20_80 50_50 80_20)

for i in 0 1 2
do
    for j in 0 1 2
    do
       for k in 10 9 8 7 6 5 4 3 2 1
       do
           echo "classifier  ${arr1[i]}"
           echo "ratio: ${arr2[j]}"
           echo "num features: $k"
           python ml_reg_top10_wo_NumPeers.py -cls ${arr1[i]} -rat ${arr2[j]} -fea $k

           
       done
    done
done