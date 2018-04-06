#!/bin/bash
gnuplot ./graph-minibatches.gplt
gnuplot -e "batch=1" ./graph-minibatches.gplt

gnuplot -e "deforms=1" ./graph-minibatches.gplt
gnuplot -e "batch=1;deforms=1" ./graph-minibatches.gplt

gnuplot -e "transforms=1" ./graph-minibatches.gplt
gnuplot -e "batch=1;transforms=1" ./graph-minibatches.gplt

gnuplot -e "transforms=1;deforms=1" ./graph-minibatches.gplt
gnuplot -e "batch=1;transforms=1;deforms=1" ./graph-minibatches.gplt

gnuplot -e "batchsize=100" ./graph-single-size.gplt
gnuplot -e "batchsize=200" ./graph-single-size.gplt
gnuplot -e "batchsize=400" ./graph-single-size.gplt
gnuplot -e "batchsize=500" ./graph-single-size.gplt
gnuplot -e "batchsize=1000" ./graph-single-size.gplt
