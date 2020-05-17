#!/bin/sh


for h in 1 2 3 4 5
do   
    for i in 1 2 3 4 
    do           
        echo "setting number is $i"
        echo "file: $h"
        python ml_svm.py -set $i -file $h

    done
done