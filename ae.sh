#!/bin/sh


# arr=(mix tanh relu)

for i in 1 2 3 4
do
    for j in mix tanh relu;
    do
       for k in 2 4 6 8
       do
           for l in 2 3 4 5 
           do
           
               echo "setting number is $i"
               echo "activation: $j"
               echo "layer: $k"
               echo "file: $l"
               python ml_ae.py -set $i -act $j -lay $k -file $l
           done
       done
    done
done