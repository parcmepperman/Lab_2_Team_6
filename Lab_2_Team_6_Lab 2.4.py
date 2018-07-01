# getting the needed library
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import neighbors,datasets

# load in dataset from library
dataset = datasets.load_iris()

# give datasets a variable name
data_x = dataset.data
data_y = dataset.target

# split the train and test
train_x, test_x, train_y, test_y=train_test_split(data_x, data_y, test_size=0.2, random_state=22)

# get the K_mean
k_mean = neighbors.KNeighborsClassifier(n_neighbors=100)

# use .kit to put the train data into the k_mean
k_mean.fit(train_x, train_y)

# k_mean prediction
predict = k_mean.predict(test_x)

# calculating the accuracy
print("Accuracy with k = 1")
print("Accuracy is:", accuracy_score(predict, test_y))