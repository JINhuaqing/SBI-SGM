singularity exec sgm_latest.sif python ../python_scripts/NSF_FOOOF.py

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
