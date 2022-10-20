singularity exec sgm_latest.sif pwd
singularity exec sgm_latest.sif ls /home/hujin/jin/*
#singularity exec sgm_latest.sif python /home/hujin/jin/MyResearch/SBI-SGM/python_scripts/NSF_FOOOF.py

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
