import math as mt

def min_max_norm(data):
    # iterate over each column (but class)
    for col in range(0, data.shape[1] - 1):
        min_val = data.iloc[:, col].min()   # min val of current col
        max_val = data.iloc[:, col].max()   # max val
        # iterate over rows
        for row in range(0, data.shape[0]):
            # normalize as per min max norm
            data.iat[row, col] = (data.iat[row, col] - min_val) / (max_val - min_val)
    return data

def distance_euclid(p, q, dim):
    # p, d ... points, dim ... dimension
    sum_squared = 0
    for i in range(dim):
        sum_squared += (p[i] - q[i]) ** 2
    return mt.sqrt(sum_squared)

def get_folds(data, k, i):
    fold_size = mt.floor(data.shape[0] / k)
    test_start = i * fold_size
    test_end = test_start + fold_size   # exclusive end index

    train = data.drop(range(test_start, test_end))
    test = data.iloc[test_start:test_end]
    X_train = train.iloc[:, :-1]
    X_test = test.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    y_test = test.iloc[:, -1]

    return [X_train, X_test, y_train, y_test]