singularity exec ~/MyResearch/sgm_latest.sif python ../python_scripts/SBI_run_mul_R1.py --noise_sd 1.6 --num_prior_sps 100000
singularity exec ~/MyResearch/sgm_latest.sif python ../python_scripts/results_process.py --noise_sd 1.6 --num_prior_sps 100000

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
