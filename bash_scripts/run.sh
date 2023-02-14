#!/bin/bash
qsub -cwd -pe smp 30 -l mem_free=2G -l h_rt=20:00:00 $1
