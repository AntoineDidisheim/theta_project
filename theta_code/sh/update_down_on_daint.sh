#!/usr/bin/env bash
# copy necessary code on the daint server
#scp {loader,parameters,ml_models,run/run1,run/run2,run/run_grid1,run/run_grid2}.py slurm/{run1,run2,grid1,grid2}.slurm sh_scripts/move_results_to_project.sh sh_scripts/move_logs_to_project.sh adidishe@ela.cscs.ch:rdm/
scp down.tar.gz kermit@130.223.173.241:theta_project/theta_code/



