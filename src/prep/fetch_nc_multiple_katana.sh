#!/bin/bash
SSH_ADDRESS='kdm.restech.unsw.edu.au'
SSH_USER='z3533092'
REMOTE_PATH='/srv/scratch/z3533092/22yr_BARRA_nt/'
LOCAL_PATH='../../data/proc/'

FILES_GLOB="22yr_barra_nt_outer_avg_*_temp_5daymn.nc 22yr_barra_nt_outer_avg_*_zeta_5daymn.nc"

for f in $FILES_GLOB
    do 
        echo "downloading ${REMOTE_PATH}/${f}"
        rsync --progress -avzh ${SSH_USER}@${SSH_ADDRESS}:${REMOTE_PATH}/${f} ${LOCAL_PATH}
    done
