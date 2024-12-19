@echo off
if exist assignment3_submission.zip del /F /Q assignment3_submission.zip
tar -a -c -f assignment3_submission.zip 01_simple_neural_network/configs 01_simple_neural_network/models 01_simple_neural_network/optimizer 01_simple_neural_network/*.py 02_optimization/code/*.ipynb 03_initialization/code/*.ipynb