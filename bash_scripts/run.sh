qsub -cwd -pe smp 50 -l mem_free=20G -l h_rt=5:00:00 job.sh 
