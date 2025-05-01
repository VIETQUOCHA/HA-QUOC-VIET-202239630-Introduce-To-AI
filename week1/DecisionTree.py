from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#Extracting features and target labels
df = pd.read_csv("./week1/iris.csv")
df_split_X = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']] 
df_split_Y = df['Name']

#data train 50%, data test 50%
X_train, X_test, y_train, y_test = train_test_split(df_split_X, df_split_Y, test_size=0.5, random_state=42)


#consider the depth of the tree to be 3 to avoid overfitting
Decision_T_Model = DecisionTreeClassifier(max_depth=3, random_state=42)
Decision_T_Model.fit(X_train, y_train)

# Ues the trained model to predict labels for the test data
y_pred = Decision_T_Model.predict(X_test)

#compare two data
accuracy = accuracy_score(y_test, y_pred)

#print last conditional sentence in the model
tree_rules = export_text(Decision_T_Model, feature_names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
print(tree_rules)
print(f"Accuracy: {accuracy * 100:.2f}%")

#Encode predicted labels into numbers for visualization
label_encoder = LabelEncoder()
y_pred_encoded = label_encoder.fit_transform(y_pred)  


plt.scatter(X_test['SepalLength'], X_test['PetalLength'], c=y_pred_encoded, cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('SVM Predictions on Iris Test Set')
plt.colorbar(label='Predicted Class')  
plt.show()