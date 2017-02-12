from sklearn import datasets
from sklearn.svm import SVC
import numpy as np

faces = datasets.fetch_olivetti_faces().data

svc_1 = SVC(kernel='linear')

from sklearn.model_selection import train_test_split

target = [1 if i % 2 == 0 else 0 for i in range(len(faces))]
target = np.array(target).astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    faces, target, test_size=0.25, random_state=0)

from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem


def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold cross validation iterator
    cv = KFold(n_splits=K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print (scores)
    print ("Mean score: {0:.3f} (+/-{1:.3f})".format(
        np.mean(scores), sem(scores)))


from sklearn import metrics


def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)

    print ("Accuracy on training set:")
    print (clf.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)

    print ("Classification Report:")
    print (metrics.classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (metrics.confusion_matrix(y_test, y_pred))

evaluate_cross_validation(svc_1, X_train, y_train, 5)

train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)
