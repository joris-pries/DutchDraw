# BinaryBaselines

BinaryBaselines is a Python package for binary classification. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the package

```bash
pip install BinaryBaselines
```

or for Windows users

```bash
python -m pip install BinaryBaselines
```



## Method
To properly assess the performance of a binary classification model, the score of a chosen measure should be compared with the score of a 'simple' baseline. E.g. an accuracy of 0.9 isn't that great if a model (without knowledge) attains an accuracy of 0.88. 

### Basic baseline
Let $\color{orange} M$  be the total number of samples. Randomly shuffle the samples and label the first $\color{orange} \lfloor \theta \cdot M \rceil$ samples as $\color{orange} 1$ and the rest as $\color{orange} 0$. This gives a baseline for each $\color{orange} \theta \in [0,1]$ . Our package can optimize (maximize and minimize) the baseline.

## Reasons to use
This package contains multiple functions. Let `true_labels` be the actual labels and `predicted_labels` be the labels predicted by a model.

If: 
* You want to use an included measure --> `measure_score(true_labels, predicted_labels, measure)`
* You want to get statistics of a baseline --> `basic_baseline_statistics(theta, true_labels, measure = possible_names)`
* You want to get statistics of the optimal baseline --> `optimized_basic_baseline(true_labels, measure = possible_names)`

### List of all included measures
|  Measure  | Definition  |
|---|:---:|
| TP |TP|
| TN | TN|
| FP | FP|
| FN | FN|
| TPR | TP / P|
| TNR | TN / N|
| FPR | FP / N|
| FNR | FN / P|
| PPV | TP / (TP + FP)|
| NPV | TN / (TN + FN)|
| FDR | FP / (TP + FP)|
| FOR | FN / (TN + FN)|
| ACC | (TP + TN) / M|
| BACC |(TPR + TNR) / 2 |
| FBETA | ((1 + β<sup>2</sup>) * TP) / ((1 + β<sup>2</sup>) * TP + β<sup>2</sup> * FN + FP)|
| MCC | (TP * TN - FP * FN) / (sqrt((TP + FP) * (TN + FN) * P * N)) |
| BM | TPR + TNR - 1|
| MK | PPV + NPV - 1|
| COHEN | (P<sub>o</sub> - P<sub>e</sub>) / (1 - P<sub>e</sub>) with P<sub>o</sub> = (TP + TN) / M and <br> P<sub>e</sub> = ((TP + FP) / M) * (P / M) + ((TN + FN) / M) * (N / M)|
| G1 | sqrt(TPR * PPV)  |
| G2 | sqrt(TPR * TNR) |
| TS | TP / (TP + FN + FP)|
| PT | (sqrt(TPR * FPR) - FPR) / (TPR - FPR)|

## Usage

As example, we first generate the true and predicted labels.
```python
import random 
random.seed(123) # To ensure similar outputs

predicted_labels = random.choices((0,1), k = 10000, weights = (0.99, 0.1))
true_labels = random.choices((0,1), k = 10000, weights = (0.99, 0.1))
```

#### Measure performance
To examine the performance of the predicted labels, we measure the markedness (MK) and F<sub>2</sub> score (FBETA).

```python
import BinaryBaselines

# Measuring markedness (MK):
print("Markedness: %6.4f" % measure_score(true_labels, predicted_labels, measure = 'MK'))

# Measuring FBETA for beta = 2:
print("F2 Score: %6.4f" % measure_score(true_labels, predicted_labels, measure = 'FBETA', beta = 2))
```
This returns as output
```python
Markedness: 0.0092
F2 Score: 0.1007
```

Note that `FBETA` is the only measure that requires additional parameter values.

#### Get basic baseline


## License
[MIT](https://choosealicense.com/licenses/mit/)