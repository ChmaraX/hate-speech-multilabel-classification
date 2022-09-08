#!/bin/bash

read -p "Enter model name: " model_name
echo $model_name

nohup python -u main.py "$model_name" > training_output.log 2>&1 &
echo $! > save_pid.txt
