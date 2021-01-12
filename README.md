![linting](https://github.com/calmdown13/dazed/workflows/Linting/badge.svg)
![tests](https://github.com/calmdown13/dazed/workflows/Tests/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/dazed.svg)](https://pypi.org/project/dazed/)

# 💫 Dazed
Dazed is little confusion matrix package designed to make your life easier. Its main advantages over alternatives (such as `sklean.metrics.confusion_matrix`) are:
- support for lots of different input formats (sparse integers, sparse strings, one-hot arrays, dataframes)
- support for multilabel annotations
- functionality to list most confused labels
- functionality to support exploration of confused samples
- prints nicely

## Installation
For the most basic installation:
```bash
pip install dazed
```
For pandas support:
```bash
pip install dazed[pandas]
```

## Basic Usage
### Sparse Data
```python
from dazed import ConfusionMatrix

truth = ["cat", "dog", "cat", "dog", "fish"]
pred = ["cat", "dog", "dog", "cat", "fish"]

ConfusionMatrix.from_sparse(truth, pred)
```
```
  | 0 1 2     index | label
---------     -------------
0 | 1 1 0         0 |   cat
1 | 1 1 0         1 |   dog
2 | 0 0 1         2 |  fish
---------     -------------
```
### One-Hot Data
```python
from dazed import ConfusionMatrix
import numpy as np

truth = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
pred = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
labels = ["cat", "dog", "fish"]

ConfusionMatrix.from_onehot(truth, pred, labels)
```
```
  | 0 1 2     index | label
---------     -------------
0 | 1 1 0         0 |   cat
1 | 1 1 0         1 |   dog
2 | 0 0 1         2 |  fish
---------     -------------
```
### DataFrame Data
```python
from dazed import ConfusionMatrix
import pandas as pd

df = pd.DataFrame({
    "cat_truth": [1, 0, 1, 0, 0],
    "dog_truth": [0, 1, 0, 1, 0],
    "fish_truth": [0, 0, 0, 0, 1],
    "cat_pred": [1, 0, 0, 1, 0],
    "dog_pred": [0, 1, 1, 0, 0],
    "fish_pred": [0, 0, 0, 0, 1]
})
truth = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
pred = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
labels = ["cat", "dog", "fish"]

ConfusionMatrix.from_df(
    df,
    ["cat_truth","dog_truth","fish_truth"],
    ["cat_pred","dog_pred","fish_pred"]
)
```
```
  | 0 1 2     index | label
---------     -------------
0 | 1 1 0         0 |   cat
1 | 1 1 0         1 |   dog
2 | 0 0 1         2 |  fish
---------     -------------
```
### Multilabel Data
```python
from dazed import ConfusionMatrix

truth = [["cat"], ["dog"], ["cat"], ["dog"], ["cat", "fish"]]
pred = [["cat"], ["dog"], ["cat", "dog"], ["cat", "dog"], ["fish"]]

ConfusionMatrix.from_sparse(truth, pred, multilabel=True)
```
```
  | 0 1 2 3 4     index |     label
-------------     -----------------
0 | 1 1 0 0 0         0 |       cat
1 | 0 0 0 0 0         1 |  cat, dog
2 | 0 0 0 0 1         2 | cat, fish
3 | 0 1 0 1 0         3 |       dog
4 | 0 0 0 0 0         4 |      fish
-------------     -----------------
```
### Most Confused
```python
from dazed import ConfusionMatrix

truth = ["cat", "dog", "cat", "dog", "fish", "cat"]
pred = ["cat", "dog", "dog", "cat", "fish", "dog"]

ConfusionMatrix.from_sparse(truth, pred).most_confused()
```
```
[('cat', 'dog', 2), ('dog', 'cat', 1)]
```
### Label Pair Info
```python
from dazed import ConfusionMatrix

truth = ["cat", "dog", "cat", "dog", "fish", "cat"]
pred = ["cat", "dog", "dog", "cat", "fish", "dog"]
sample_info = ["image0.jpg", "image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg"]

cm = ConfusionMatrix.from_sparse(truth, pred, info=sample_info)
cm.label_pair_info("cat", "dog")
```
```
['image2.jpg', 'image5.jpg']
```
```python
cm.label_pair_info("dog", "cat")
```
```
['image3.jpg']
```
```python
cm.label_pair_info("cat", "fish")
```
```
[]
```
### As Array
```python
from dazed import ConfusionMatrix

truth = ["cat", "dog", "cat", "dog", "fish", "cat"]
pred = ["cat", "dog", "dog", "cat", "fish", "dog"]

ConfusionMatrix.from_sparse(truth, pred).as_array()
```
```
(array([[1, 2, 0],
        [1, 1, 0],
        [0, 0, 1]]),
 ['cat', 'dog', 'fish'])
```
### As DataFrame
```python
from dazed import ConfusionMatrix

truth = ["cat", "dog", "cat", "dog", "fish", "cat"]
pred = ["cat", "dog", "dog", "cat", "fish", "dog"]

ConfusionMatrix.from_sparse(truth, pred).as_df()
```
|| cat | dog | fish |
|:-|-:|-:|-:|
| cat | 1 | 2 | 0 |
| dog | 1 | 1 | 0 |
| fish | 0 | 0 | 1 |
