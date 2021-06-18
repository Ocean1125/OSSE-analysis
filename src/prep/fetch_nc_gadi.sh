#!/bin/zsh
SSH_ADDRESS='gadi-dm.nci.org.au'
SSH_USER='deg581'
#REMOTE_PATH='/g/data/fu5/deg581/EAC_2yr_OSSE_SSHSST/output/'
REMOTE_PATH='/g/data/fu5/deg581/EAC_2yr_truthRun_obsVerification_HighRes/output/'
#FILE_GLOB='roms_fwd_outer*_0800*.nc'
FILE_GLOB='outer_his_0800*.nc'
LOCAL_PATH='../../data/raw/'

rsync --progress -avzh ${SSH_USER}@${SSH_ADDRESS}:${REMOTE_PATH}/${FILE_GLOB} ${LOCAL_PATH}
