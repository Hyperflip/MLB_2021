from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from MySmote import MySmote
from util import plot_classes

# define dataset
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# smote implementation from imbalance-learn
smote = SMOTE()
X_1, y_1 = smote.fit_resample(X, y)

# my smote implementation
mySmote = MySmote(k=5)
X_2, y_2 = mySmote.fit_resample(X, y)

plot_classes(X, y, 'imbalanced data')
plot_classes(X_1, y_1, 'resampled by SMOTE from imbalance-learn')
plot_classes(X_2, y_2, 'resampled by custom SMOTE implementation')
