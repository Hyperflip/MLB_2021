import pandas as pd
import math as mt
from util import min_max_norm, get_folds
from myKNN import myKNNClassifier
from sklearn.neighbors import KNeighborsRegressor

if __name__ == '__main__':
    # data loading
    data = pd.read_excel('data/real_estate.xlsx').drop(columns='No')
    # normalize data
    data = min_max_norm(data)
    # shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    k = 3
    num_folds = 5
    mean_rmsds_per_k_myKNN = []
    mean_rmsds_per_k_sklearn = []

    # hyperparam tuning with odd values of k
    for k in range(1, 10, 2):
        rmsds_sum_per_kFold_myKNN = 0
        rmsds_sum_per_kFold_sklearn = 0
        # do num_folds amount of splits
        for i in range(0, num_folds):
            # get random split
            folds = get_folds(data, num_folds, i)
            X_train = folds[0]
            X_test = folds[1]
            y_train = folds[2]
            y_test = folds[3]

            # fit regressors
            regressor_myKNN = myKNNClassifier(k, 'regressor', X_train, y_train)
            regressor_sklearn = KNeighborsRegressor(k)
            regressor_sklearn.fit(X_train, y_train)

            total = X_test.shape[0]
            sum_squared_myKNN = 0
            sum_squared_sklearn = 0
            for X, y in zip(X_test.values.tolist(), y_test.values.tolist()):
                diff_squared_myKNN = (regressor_myKNN.predict(X) - y) ** 2
                diff_squared_sklearn = (regressor_sklearn.predict([X]) - y) ** 2

                sum_squared_myKNN += diff_squared_myKNN
                sum_squared_sklearn += diff_squared_sklearn

            rmsd_per_kFold_myKNN = mt.sqrt(sum_squared_myKNN / total)
            rmsd_per_kFold_sklearn = mt.sqrt(sum_squared_sklearn / total)

            rmsds_sum_per_kFold_myKNN += rmsd_per_kFold_myKNN
            rmsds_sum_per_kFold_sklearn += rmsd_per_kFold_sklearn

        mean_rmsds_per_k_myKNN.append(round(rmsds_sum_per_kFold_myKNN / num_folds, 2))
        mean_rmsds_per_k_sklearn.append(round(rmsds_sum_per_kFold_sklearn / num_folds, 2))

    print(mean_rmsds_per_k_myKNN)
    print(mean_rmsds_per_k_sklearn)