#! /bin/bash

#SBATCH -p cbuild
#SBATCH -t 04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.evers-2@student.tudelft.nl

srun find -O3 /gpfs/scratch1/nodespecific/ -maxdepth 2 -type d -user cmeo -exec rm -rf {} \;
