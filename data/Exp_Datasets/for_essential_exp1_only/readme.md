# Data Description

Using pseudo random seed to generate and split datasets for the experiment. 
The seed is set based on the date and time of a running of the code. 
The seed info is added at the end of the folder name of the dataset. 
e.g. `Q2_Datasets_rs2310092123` means 2310092123 is the seed. 

In order to avoid information leakage, an exhaust training process will be done 
using the same dataset (same seed) in order to ensure the validity of the training results. 

For a specific question, five datasets are generated using five different seeds. 
We will experiment on each dataset to get the average performance. 
This is to remove the possible bias of a single shuffle.

Under each dataset folder, there are three groups of data:

- 'x_train.npy' and 'y_train.npy' are the training data and labels.
- 'x_test.npy' and 'y_test.npy' are the testing data and labels.
- 'x_val.npy' and 'y_val.npy' are the validation data and labels.

Usage:

```python
import numpy as np
x1_train, x2_train = np.load('x_train.npy') 
y_train = np.load('y_train.npy')
```

note: x_train shape: (2, num_samples, img_size, img_size, 3), y_train shape: (num_samples,) 
for ranking task







