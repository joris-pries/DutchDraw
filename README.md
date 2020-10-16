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

#### Base measures:
* TP:
* TN:
* FP:
* FN:

##### Other measures
* TPR: True Positive Rate. Defined as $\frac{TP}{P}$
* TNR:
* FPR:
* FNR:
* PPV:
* NPV:
* FDR:
* FOR:
* ACC:
* BACC:
* FBETA:
* MCC:
* BM:
* MK:
* COHEN:
* G1:
* G2:
* G2 APPROX:
* FOWLKES:
* TS:
* PT:


|  Measure  | Definition  |
|---|:---:|
| TP | <img src="https://render.githubusercontent.com/render/math?math=\color{orange} TP">|
| TN | $\color{orange} TN$|
| FP | $\color{orange} FP$|
| FN | $\color{orange} FN$|
| TPR | $\color{orange} \frac{TP}{P}$|
| TNR | $\color{orange} \frac{TN}{N}$|
| FPR | $\color{orange} \frac{FP}{N}$|
| FNR | $\color{orange} \frac{FN}{P}$|
| PPV | $\color{orange} \frac{TP}{TN + FN}$|
| NPV | $\color{orange} $|
| FDR | $\color{orange} $|
| FOR | $\color{orange} $|
| ACC | $\color{orange} $|
| BACC | $\color{orange} $|
| FBETA | $\color{orange} $|
| MCC | $\color{orange} $|
| BM | $\color{orange} $|
| MK | $\color{orange} $|
| COHEN | $\color{orange} $|
| G1 | $\color{orange} $|
| G2 | $\color{orange} $|
| G2 APPROX | $\color{orange} $|
| FOWLKES | $\color{orange} $|
| TS | $\color{orange} $|
| PT | $\color{orange} $|

## Usage


```python
import BinaryBaselines

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```



## License
[MIT](https://choosealicense.com/licenses/mit/)