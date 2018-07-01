from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics

# setting variables
x = datasets.load_digits()
info = x.data
y = x.target

# this is where we will split the info to start testing
X_train, X_test,y_train,y_test = train_test_split(info, y, test_size=0.2, random_state=1)

# this is where we split train to start the RBF
X_train_rbf, X_test_rbf, y_train_rbf, y_test_rbf = train_test_split(info, y, test_size=0.2, random_state=1)

# giving the linear kernel a name
model = SVC(kernel='linear')

# giving the RBF kernel a name
rbf = SVC(kernel='rbf')

# the train in linear
model.fit(X_train, y_train)

# predicting the linear
linear_pre = model.predict(X_test)

# computing the accuracy for linear
print("linear Accuracy:", metrics.accuracy_score(y_test, linear_pre) * 100)

# the train in RBF
rbf.fit(X_train_rbf, y_train_rbf)

# predicting the RBF
rbf_prediction = rbf.predict(X_test_rbf)

print("RBF Accuracy:", metrics.accuracy_score(y_test_rbf, rbf_prediction) * 100)