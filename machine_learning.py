import pickle

from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
import os
import numpy as np

data_dir = 'lights_train'

fnames = []
for fname in os.listdir(data_dir):
    if fname.startswith('file'):
        fnames.append(fname)

raw_data = [misc.imread(os.path.join(data_dir, fname)) for fname in fnames]
img_dim = (90, 90)
data = np.zeros(((len(raw_data), img_dim[0] * img_dim[1])), dtype=np.float64)
for img_i in range(len(raw_data)):
    print('img {}'.format(img_i))
    for i in range(data.shape[1]):
        row = i / raw_data[img_i].shape[0]
        col = i % raw_data[img_i].shape[1]
        if row < raw_data[img_i].shape[0] and col < raw_data[img_i].shape[1]:
            # average rgb channels to single value
            avg = (raw_data[img_i][row][col][0] / 3.0 + raw_data[img_i][row][col][1] / 3.0 + raw_data[img_i][row][col][2] / 3.0)
            data[img_i][i] = avg / 255.0

faces = datasets.fetch_olivetti_faces().data

from sklearn import svm

X = data
y = [1 if fname.endswith('True.png') else 0 for fname in fnames]


def train():
    clf = svm.SVC()
    clf.fit(X, y)
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=True)

    pickle.dump(clf, open('svc.model', 'w'))

    print ("Accuracy on training set:")
    print (clf.score(X, y))


def predict():
    clf2 = pickle.load(open('svc.model'))
    Y_predict = clf2.predict(X)
    print(Y_predict)


train()

predict()
