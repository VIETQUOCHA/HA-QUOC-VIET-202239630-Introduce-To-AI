import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#Extracting features and target labels
df = pd.read_csv("./week1/iris.csv")
X = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']] 
y = df['Name']

#data train 70%, data test 30%
X_train, X_test, Y_train, Y_test = train_test_split(X,y,random_state=42,test_size=0.3)

#train model and set number of trees = 100
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train,Y_train)

# Ues the trained model to predict labels for the test data
rf_pred = rf_model.predict(X_test)

#accuracy of model
rf_accuracy = accuracy_score(Y_test,rf_pred)

# Print  conditional sentence in the model of tree 1
tree_rules1 = export_text(rf_model.estimators_[0], feature_names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
print("Tree 1:" +"\n"+tree_rules1)

print("----------------------------------")

# Print  conditional sentence in the model of tree 2
tree_rules2 = export_text(rf_model.estimators_[1], feature_names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
print("Tree 2:" +"\n"+tree_rules2)

print(f"accuracy: {rf_accuracy * 100:.2f}%") 

#Encode predicted labels into numbers for visualization
label_encoder = LabelEncoder()
y_pred_encoded = label_encoder.fit_transform(rf_pred) 


plt.scatter(X_test['SepalLength'], X_test['PetalLength'], c=y_pred_encoded, cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('SVM Predictions on Iris Test Set')
plt.colorbar(label='Predicted Class')  
plt.show()