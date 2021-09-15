from util import distance_euclid

class myKNNClassifier:
    def __init__(self, k, type, X_train, y_train):
        self.k = k
        self.type = type
        self.X_train = X_train
        self.y_train = y_train
        self.dim = X_train.shape[1]

    def predict(self, X_predict):
        distances = []
        # calc distances
        for X, y in zip(self.X_train.values.tolist(), self.y_train.values.tolist()):
            distance = distance_euclid(X, X_predict, self.dim)
            distances.append((distance, y))
        # sort ascending
        distances.sort(key=lambda x: x[0])

        if self.type == 'classifier':
            # classify by knn
            count = {}
            # init count for each class
            for class_name in set(self.y_train):
                count[class_name] = 0
            for i in range(self.k):
                count[distances[i][1]] += 1
            return max(count, key=count.get)
        elif self.type == 'regressor':
            # gather y by knn
            sum = 0
            for i in range(self.k):
                sum += distances[i][1]
            # return mean
            return sum / self.k
