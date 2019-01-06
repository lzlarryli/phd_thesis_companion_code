#!/bin/bash
python exp_convergence.py 0 64 5
python exp_convergence.py 1 32 5
python exp_convergence.py 2 32 5
python exp_convergence.py 3 16 5
python exp_long_time_behavior.py 0 1024 100
python exp_long_time_behavior.py 1 128 300
python exp_long_time_behavior.py 2 64 300
python exp_long_time_behavior.py 3 32 300

python exp_long_time_behavior.py 1 256 3000
python exp_long_time_behavior.py 2 128 3000

