#!/bin/bash

mkdir -p outputs

echo "Run all the Exercises"
echo "----------------------"
echo -e "Running exercise 1..."
./01-code-submissions.py

echo -e "\nRunning exercise 2..."
./02-code-submissions.py

echo -e "\nRunning exercise 3..."
./03-code-submissions.py

echo -e "\nSCORES"
echo "----------------------"
echo -e "\nScore for exercise 1..."
python score.py prediction outputs/01_output_validation.csv outputs/01_output_prediction_1239.csv

echo -e "\nScore for exercise 2..."
python score.py balance data/02_input_target.csv data/02_input_capacity.csv outputs/02_output_productionPlan_1239.csv outputs/02_output_shipments_1239.csv

echo -e "\nScore for exercise 3..."
python score.py cost data/02_input_target.csv data/02_input_capacity.csv data/03_input_productionCost.csv data/02_03_input_shipmentsCost.csv outputs/03_output_productionPlan_1239.csv outputs/03_output_shipments_1239.csv