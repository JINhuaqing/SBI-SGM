singularity exec ~/MyResearch/sgm_latest.sif python ../python_scripts/results_process.py --noise_sd 1.2 --num_prior_sps 200000

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
