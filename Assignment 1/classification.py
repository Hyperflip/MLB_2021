import pandas as pd
from util import min_max_norm, get_folds
from myKNN import myKNNClassifier
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    # data loading
    data_headers = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    data = pd.read_csv('data/iris.data', names=data_headers)
    # normalize data
    data = min_max_norm(data)
    # shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    k = 3
    num_folds = 5
    mean_accuracies_myKNN = []
    mean_accuracies_sklearn = []

    # hyperparam tuning with odd values of k
    for k in range(1, 10, 2):
        accuracy_sum_myKNN = 0
        accuracy_sum_sklearn = 0
        # do num_folds amount of splits
        for i in range(0, num_folds):
            # get random split
            folds = get_folds(data, num_folds, i)
            X_train = folds[0]
            X_test = folds[1]
            y_train = folds[2]
            y_test = folds[3]

            # fit classifiers
            classifier_myKNN = myKNNClassifier(k, 'classifier', X_train, y_train)
            classifier_sklearn = KNeighborsClassifier(k)
            classifier_sklearn.fit(X_train, y_train)

            total = X_test.shape[0]
            correct_myKNN = 0
            correct_sklearn = 0
            for X, y in zip(X_test.values.tolist(), y_test.values.tolist()):
                if classifier_myKNN.predict(X) == y:
                    correct_myKNN += 1
                if classifier_sklearn.predict([X]) == y:
                    correct_sklearn += 1

            accuracy_myKNN = correct_myKNN / total
            accuracy_sklearn = correct_sklearn / total

            accuracy_sum_myKNN += accuracy_myKNN
            accuracy_sum_sklearn += accuracy_sklearn

        mean_accuracies_myKNN.append(round(accuracy_sum_myKNN / num_folds, 2))
        mean_accuracies_sklearn.append(round(accuracy_sum_sklearn / num_folds, 2))

    print(mean_accuracies_myKNN)
    print(mean_accuracies_sklearn)

