#!/bin/bash

until ~/miniconda3/envs/whatforwhere/bin/python ~/repos/what-for-where/scripts/fetch/get_random_sample.py; do
    echo "Script crashed with exit code $?. Respawning.." >&2
    sleep 1
done   
