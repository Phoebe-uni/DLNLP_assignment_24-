# DLNLP_assignment_24-SN23201000

This project using the BERT and RoBERTa model to complete the task of Tweet Sentiment Extraction on Kaggle competition [here]([https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://www.kaggle.com/competitions/tweet-sentiment-extraction/data). 

## 1. Prerequisites
To Begin with, it is required to download the datasets and put it into the nlp2/Datasets folder. 

The folder Datasets is not included, you can either download the required Datesets according to the file tree or you can download the zip file [here]([https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://www.kaggle.com/competitions/tweet-sentiment-extraction/data). 

### The environment

My advice is to create a new conda environment by the following command first:

```bash
conda create -n myenv python=3.10.14
```
Then, enter the nlp2 folder, activate the environment created above and install all the package that required for this project by:

```bash
pip install -r requirements.txt
```

## 2. How to check the result of this project

Just run the main.py

## 3. Explanation of folder structure

The BERT model is in the folder A, the output folder includes the output of the model. The input folder includes the parameter file that need for data training.
The RoBERTa model is in the folder B, the output folder includes the output of the model. The input folder includes the parameter file that need for data training.
The output of those two models are also included in the Datasets folder.
