import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.DataFrame({
    'Age': [18,20,22,25,28,30,35,40,45,50],
    'Purchase': [0,0,0,0,1,1,1,1,1,1]
})

X, Y = df[['Age']], df['Purchase']
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, Y)

age = float(input("Enter Customer Age: "))
user = pd.DataFrame({'Age': [age]})

pred = model.predict(user)[0]
prob = model.predict_proba(user)[0]

print("Not Purchase:", round(prob[0]*100, 2), "%")
print("Purchase:", round(prob[1]*100, 2), "%")
print("Prediction:", "WILL PURCHASE" if pred else "WILL NOT PURCHASE")

# Simple decision tree graph
plot_tree(model, feature_names=['Age'],
          class_names=['No', 'Yes'], filled=True)
plt.show()