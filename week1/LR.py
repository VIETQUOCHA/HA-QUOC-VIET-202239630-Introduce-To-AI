import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#Extracting features and target labels
df = pd.read_csv("./week1/iris.csv")
X = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']] 
y = df['Name']

#data train 80%, data test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Train the model with a maximum of 200 iterations for convergence
lr_model = LogisticRegression(max_iter=200)  
lr_model.fit(X_train, y_train)

# Ues the trained model to predict labels for the test data
y_pred = lr_model.predict(X_test)

#compare two data
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy: {accuracy * 100:.2f}%")

#Encode predicted labels into numbers for visualization
label_encoder = LabelEncoder()
y_pred_encoded = label_encoder.fit_transform(y_pred)  


plt.scatter(X_test['SepalLength'], X_test['PetalLength'], c=y_pred_encoded, cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('SVM Predictions on Iris Test Set')
plt.colorbar(label='Predicted Class')  
plt.show()