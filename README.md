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

## Roadmap

Iris classification, MNIST digit recognition, movie review sentiment, customer churn, decision tree, PCA, Markov chain text generator, neural network and linear regression from zero.

## How to run

```bash
pip install -r requirements.txt
cd 01-house-price-prediction
python main.py
```

---

**Author:** Tomás Alarcón, Systems and Computing Engineering, Universidad de los Andes
