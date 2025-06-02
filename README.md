# Real-Time Anomaly Segmentation for Road Scenes

This repository contains the implementation of our project for the Advanced Machine Learning course.
The complete project methodology, experimental results, and discussion are documented in the attached [project report](./Paper-Bouchari-Bergamini-Rabellino.pdf).

## Structure

- [eval](/eval/): contains scripts for evaluating trained models, inclutiong various scoring methods (MSP, MaxLogit, MaxEntropy, Void classification) and code for model pruning and quantization.
- [train](/train/): contains scripts for training the models, including functionality specifically for the void classifier, and model definitions.
- [trained_models](/trained_models/): includes pre-trained models.

## Datasets

The datasets used for training (Cityscapes) evaluation (Road Anomaly, Road Obstacles and Fishyscapes) must be uploaded to the user's Google Drive. The notebook expects the images to be located under /content/drive/MyDrive/... with specific subfolder names, as referenced in the code. Some results may also be saved to the same Drive location during execution.

## Notebook reference

All code, experiments, and result visualizations are integrated in the [Jupyter notebook](/RealtimeAnomalySegmentation.ipynb), which serves as the main interface for reproducing the project pipeline. It references modules and scripts in the repository.
