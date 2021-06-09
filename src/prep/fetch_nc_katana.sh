#!/bin/zsh
SSH_ADDRESS='kdm.restech.unsw.edu.au'
SSH_USER='z3533092'
REMOTE_PATH='/srv/scratch/z3533092/22yr_BARRA_nt/'
FILE_GLOB='22yr_barra_nt_outer_avg_*_temp_5daymn.nc'
LOCAL_PATH='../../data/proc/'

rsync --progress -avzh ${SSH_USER}@${SSH_ADDRESS}:${REMOTE_PATH}/${FILE_GLOB} ${LOCAL_PATH}
