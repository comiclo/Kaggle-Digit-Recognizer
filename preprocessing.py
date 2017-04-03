import pandas as pd
import numpy as np

def load_data():
    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')

    train = train.take(np.random.permutation(len(train)))

    num_labels = 10

    def reformat(data, labels=None):
        data = data.reshape((-1, 28, 28, 1)).astype(np.float32)
        try:
            labels = labels.astype(np.float32)
            labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
            return data, labels
        except:
            return data

    # train_size = 40000
    # train_data = train.iloc[:train_size, 1:].as_matrix()
    # train_labels = train.iloc[:train_size, 0].as_matrix()
    # valid_data = train.iloc[train_size:, 1:].as_matrix()
    # valid_labels = train.iloc[train_size:, 0].as_matrix()

    # test_data = test.as_matrix()
    # train_data, train_labels = reformat(train_data, train_labels)
    # valid_data, valid_labels = reformat(valid_data, valid_labels)
    # test_data = reformat(test_data)

    train_data = train.iloc[:, 1:].as_matrix()
    train_labels = train.iloc[:, 0].as_matrix()

    test_data = test.as_matrix()
    train_data, train_labels = reformat(train_data, train_labels)
    test_data = reformat(test_data)

    del train
    del test

    return train_data, train_labels, test_data