#!/bin/bash

args="$@"

function itermini {
    for i in {100..1000..100}; do
        th ./src/main.lua $args --minibatch $i --deforms 10
        th ./src/main.lua $args --minibatch $i --transforms 10
        th ./src/main.lua $args --minibatch $i --deforms 5 --transforms 5
    done 
}

function iterdeform {
    for i in {1..10000..500}; do
        for j in {5..50..5}; do
            i_=$(bc -l <<< "$i/1000")
            th ./src/main.lua $args --minibatch 250 --deforms 10 $i_ $j 
        done
    done 
}

function iterlearn {
    for i in {1..10..1}; do
        for j in {25..150..25}; do
            i_=$(bc -l <<< "$i/10")
            th ./src/main.lua $args --minibatch 250 --deforms 10 --learning_multiplier $i_ $j
        done
    done 
}

itermini
#iterlearn
#iterdeform

args="$args --batchnorm"
echo "$args"
itermini
#iterlearn
#iterdeform
