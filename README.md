# Machine Learning from Scratch

![Python](https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white)

A series of machine learning projects where the core algorithm is **implemented by hand in NumPy**, applied to a real dataset and then benchmarked against the standard library. The goal is to understand every model well enough to derive it, debug it and explain it to a non-technical audience.

Each project lives in its own folder with its code, data, charts, interactive dashboard and README.

## Projects

| # | Project | Algorithm built from scratch | Key result |
|---|---|---|---|
| 01 | [House Price Prediction](01-house-price-prediction) | OLS (normal equation), Ridge (closed form and gradient descent) | 5.3% median error on 2,925 home sales; matches scikit-learn |
| 02 | [Iris Flower Classification](02-iris-classification) | k-NN, stratified k-fold, confusion matrix and F1 | 95.9% CV accuracy; 100% agreement with scikit-learn |
| 03 | [MNIST Digit Recognition](03-mnist-digit-recognition) | Dense and convolutional layers, backpropagation, Adam | 98.82% test accuracy with a CNN built in NumPy |
| 04 | [Movie Review Sentiment](04-movie-review-sentiment) | Text preprocessing, TF-IDF, logistic regression with Adam | 90.97% accuracy on 10,000 IMDB reviews; live demo in the browser |
| 05 | [Customer Churn Prediction](05-customer-churn-prediction) | SMOTE, stratified split, profit curve, permutation importance (with LightGBM) | AUC 0.863; profit-based threshold earns 47% more than the default |
| 06 | [Decision Tree from Scratch](06-decision-tree-from-scratch) | CART with Gini, entropy, information gain and cost-complexity pruning | Identical to scikit-learn; 6-leaf tree classifies 93% of unseen penguins |
| 07 | [PCA from Scratch](07-pca-from-scratch) | Covariance, Jacobi eigen-decomposition, power iteration, Marchenko-Pastur | The market explains 32% of 56 stocks; 9 components beat noise |

## Roadmap

Markov chain text generator, neural network and linear regression from zero.

## How to run

```bash
pip install -r requirements.txt
cd 01-house-price-prediction
python main.py
```

---

**Author:** Tomás Alarcón, Systems and Computing Engineering, Universidad de los Andes
