singularity exec ~/MyResearch/sgm_latest.sif python ../python_scripts/SBI_run.py

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
