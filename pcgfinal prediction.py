from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import pickle
tqdm.pandas()

with open('knn_classifier_test.pkl', 'rb') as f:
    knn_classifier = pickle.load(f)

with open('x_test_test.pkl', 'rb') as f:
    x_test = pickle.load(f)

with open('y_test_test.pkl', 'rb') as f:
    y_test = pickle.load(f)


y_pred_knn = knn_classifier.predict(x_test.tolist())
print(y_pred_knn[0:20])
ac_knn = accuracy_score(y_test, y_pred_knn)
print("KNN model accuracy is", ac_knn)

print(confusion_matrix(y_test, y_pred_knn))

