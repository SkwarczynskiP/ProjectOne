from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn import naive_bayes

# Section 1: Load the Iris Dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(iris_df.head())
print(iris_df.tail())

# Section 2: Visualize the Data
sns.pairplot(iris_df, height=1.5, hue='target')
plt.show()

# Section 3: Split the Data into Training and Testing Sets
iris_train_features, iris_test_features, iris_train_target, iris_test_target = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42)
print("Train Features Shape: ", iris_train_features.shape)
print("Test Features Shape: ", iris_test_features.shape)

# Section 4: Build and Evaluate the kNN Model
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(iris_train_features, iris_train_target)
knn_predict = knn.predict(iris_test_features)
knn_f1_score = metrics.f1_score(iris_test_target, knn_predict, average='weighted')

# Section 5: Build and Evaluate the Naive Bayes Model
gnb = naive_bayes.GaussianNB()
gnb.fit(iris_train_features, iris_train_target)
gnb_predict = gnb.predict(iris_test_features)
gnb_f1_score = metrics.f1_score(iris_test_target, gnb_predict, average='weighted')

# Section 6: Compare the Model Performance using the F1 Score
print("kNN F1 Score: ", knn_f1_score)
print("Gaussian Naive Bayes F1 Score: ", gnb_f1_score)
