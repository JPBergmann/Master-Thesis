# Master-Thesis
Codebase for my Master Thesis from MIBA 2023

Objective: Create a stock portfolio which (1) outperforms the market and (2) shows strong technical performance using DL.

Approach: CNN-LSTM based models
- Iteratively for each company
- Collectively trying to predict all companies with one large model

Data preprocessing is in `etl`
Model Architectures can be found in `models`
Helper functions to sequentialize tensors for 1D and 2D use can be found in `helpers`

Jupyter Notebook containing `baseline` show first experiments with data and models.
Python scripts ending with `trainer` show further experimentation with designed model architectures found in `models`.
Python scripts starting with `hp_opt` show hyperparameter optimization process.
Jupyter Notebook containing `result_eval` show final models and result metrics, also presented in the thesis paper.
