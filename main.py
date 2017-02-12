import pickle

from sklearn import datasets
from sklearn.svm import SVC

faces = datasets.fetch_olivetti_faces().data

from sklearn import svm

X = faces
y = [i % 2 for i in range(len(faces))]


def train():
    clf = svm.SVC()
    clf.fit(X, y)
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        max_iter=-10, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

    pickle.dump(clf, open('svc.model', 'w'))


def predict():
    clf2 = pickle.load(open('svc.model'))
    Y_predict = clf2.predict(X)
    print(Y_predict)


train()
