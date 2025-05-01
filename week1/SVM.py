import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#Extracting features and target labels
df = pd.read_csv("./week1/iris.csv")
X = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']] 
y = df['Name']

#data train 80%, data test 20%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=42)


# Initialize the SVM model with a linear kernel, regularization parameter C=1.0, and fixed seed for reproducibility
svm_model = SVC(kernel='linear', C=1.0, random_state=42)


# Train the SVM model on the training data to find the optimal hyperplane separating the classes
svm_model.fit(X_train,y_train)


# Use the trained model to predict labels for the test data
y_pred = svm_model.predict(X_test)


#accuracy of model
accuracy = accuracy_score(y_pred, y_test)


#Encode predicted labels into numbers for visualization
label_encoder = LabelEncoder()
y_pred_encoded = label_encoder.fit_transform(y_pred)  


plt.scatter(X_test['SepalLength'], X_test['PetalLength'], c=y_pred_encoded, cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('SVM Predictions on Iris Test Set')
plt.colorbar(label='Predicted Class')  
plt.show()
print(f"accuracy: {round(accuracy * 100, 2)}%")