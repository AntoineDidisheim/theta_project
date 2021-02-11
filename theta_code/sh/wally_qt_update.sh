#!/usr/bin/env bash
# copy necessary code on the daint server
#scp {loader,parameters,ml_models,run/run1,run/run2,run/run_grid1,run/run_grid2}.py slurm/{run1,run2,grid1,grid2}.slurm sh_scripts/move_results_to_project.sh sh_scripts/move_logs_to_project.sh adidishe@ela.cscs.ch:rdm/
scp qt_cpp/*.cpp adidishe@wally-front1.unil.ch:/scratch/wally/FAC/HEC/DF/sscheid1/default/qt_cpp
scp qt_cpp/slurm/* adidishe@wally-front1.unil.ch:/scratch/wally/FAC/HEC/DF/sscheid1/default/qt_cpp/slurm

