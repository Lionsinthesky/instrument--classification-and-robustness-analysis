from sklearn.svm import SVC

def train_svm(x_train_reshape, y_train):
    svm = SVC(verbose=True)
    svm.fit(x_train_reshape, y_train)
    return svm

def predict_svm(svm, x_test_reshape):
    return svm.predict(x_test_reshape)


