#!/usr/bin/env bash
# just scp the results and logs file from daint back to the machine for analysis
scp -r adidishe@ela.cscs.ch:/project/s885/jf_option/\{logs_daint,model_save\} /media/antoine/ssd_ntfs1/jf_option/
