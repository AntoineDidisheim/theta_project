DATA DOWNLOADING: 
- get ticker of firm that did belong to S&P500  --> run get_tic.py for that
- dowload the crps monthly with that --> save it in raw/crspy_monthly.csv 
- run Data().get_list_permno_from_crsp() to get the list of corresponding PERMNO
- use the permno from this to download the merge with optionmetric list from Option Metrics CRSP Link (Beta) --> save it in raw/crsp_to_opt.csv
- run Data().get_list_secid_from_crsp() to get list of secid
- use the list of secid form that to download the opitons again --> save this in raw/opt.csv
- download the full version of the compustat data-base from the "CRSP/Compustat Merged Database". Include columns "Historical CRSP PERMNO Link to COMPUSTAT record" --> save in raw/compustat_quarterly and raw/compustat_yearly respectively
-finally use Fama library to download all ff and put in raw

DATA pre-rpocessing
create a Data object data and run, data.pre_process_all()

DATA creation (with specific funcitonalities)
When creating a new dataset just run data.pre_process_sample()