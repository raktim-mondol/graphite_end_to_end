#!/bin/bash   
echo "Job Started @ `date`"

echo "Running Jobs"
cd $HOME

module load python3/3.9.2
module load cuda/11.7.0



echo "Load Python Environment named Image"
source /g/data/nk53/rm8989/software/image2/bin/activate



cd /scratch/nk53/rm8989/gene_prediction/code/GRAPHITE/final_code_step_2_part_2/testing/
python main_visualization.py

    
    