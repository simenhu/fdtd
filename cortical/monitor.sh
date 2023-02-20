#!/bin/sh 
#tmux new-session -s 'train' -d 'python resonate_all.py --load-step latest'
tmux new-session -s 'train' -d 'watch -n 0.1 nvidia-smi'
tmux split-window -v 'htop'
tmux rename-window 'Monitor'
tmux set -g remain-on-exit on
tmux new-window 'tensorboard --logdir=./runs/ --port=12345'
#tmux select-window -t 0
tmux -2 attach-session -d 

