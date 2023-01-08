#!/bin/bash

START_TIME=$(date +%s)
TIME_ARG=" --start_time $START_TIME"

rm logs/cpu.txt
rm logs/time.txt
rm logs/gpu.csv

PY_COMMAND="$1$TIME_ARG"
$PY_COMMAND &
pid=$!

while kill -0 $pid 2>/dev/null; do
    echo $(($(date +%s) - $START_TIME)) >> logs/time.txt &
    nvidia-smi dmon -c 1 >> logs/gpu.csv
    ps -C python3 -o %cpu >> logs/cpu.txt
    sleep 1
done

python3 usage_vis.py