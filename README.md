# Healthcare-Claims-Analysis-with-LLMs

# Healthcare Claims with LLM Embeddings

## Overview
This project applies Large Language Model (LLM) embeddings to unstructured healthcare insurance claim texts. The objective is to evaluate how dimensionality reduction techniques affect efficiency and predictive performance while preserving semantic information.

## Motivation
Healthcare claims data contains both structured codes and unstructured text. LLM embeddings provide a way to represent this text in high-dimensional vector space, capturing semantic patterns across diagnoses, procedures, and providers. However, full-size embeddings (768 dimensions) can be computationally expensive. This project investigates whether reducing embedding size can improve efficiency while maintaining accuracy.

## Methods
- Constructed synthetic unstructured claim text by combining fields such as procedure codes, diagnosis codes, and provider specialties  
- Generated embeddings using BERT and DistilBERT models  
- Applied dimensionality reduction methods (PCA, UMAP) to visualize and analyze embeddings  
- Trained a Logistic Regression classifier to predict claim type from embeddings of varying sizes (128, 256, 512, 768)  
- Measured accuracy and F1 score to evaluate trade-offs between efficiency and performance  

## Results
- PCA and UMAP visualizations showed that embeddings form meaningful clusters of similar claims  
- Reducing embeddings to 256 dimensions preserved most predictive performance compared to 768 dimensions  
- Dimensionality reduction reduced memory use and training time with minimal accuracy loss  

## Key Takeaways
- LLM embeddings effectively capture semantic relationships in healthcare-finance claims  
- Dimensionality reduction enables more efficient storage and computation without significant performance degradation  
- These methods support practical applications such as fraud detection, claim categorization, and cost prediction    

## Dataset
Enhanced Health Insurance Claims Dataset (Kaggle):  
https://www.kaggle.com/datasets/leandrenash/enhanced-health-insurance-claims-dataset  

## Requirements
- Python 3.8 or later  
- transformers  
- torch  
- scikit-learn  
- umap-learn  
- pandas, matplotlib, seaborn  

## How to Run
1. Clone the repository  
2. Install dependencies  
3. Run the notebook in Google Colab or locally  
4. Use a Kaggle API key (`kaggle.json`) to download the dataset
