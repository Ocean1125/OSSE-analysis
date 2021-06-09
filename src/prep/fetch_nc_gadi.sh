#!/bin/zsh
SSH_ADDRESS='gadi-dm.nci.org.au'
SSH_USER='deg581'
REMOTE_PATH='/g/data/fu5/deg581/EAC_2yr_OSSE_SSHSST/output/'
FILE_GLOB='roms_fwd_outer0_0800*.nc'
LOCAL_PATH='../../data/raw/'

rsync --progress -avzh ${SSH_USER}@${SSH_ADDRESS}:${REMOTE_PATH}/${FILE_GLOB} ${LOCAL_PATH}
