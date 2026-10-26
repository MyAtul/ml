# Install required libraries (run in terminal / command prompt):
# pip install numpy matplotlib scikit-learn
#py -3.11 -m pip install numpy matplotlib scikit-learn

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = load_iris()
X, y = data.data[:100, :2], data.target[:100]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel='linear').fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Hyperplane and margins
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            facecolors='none', edgecolors='red', s=100)

w = model.coef_[0]
b = model.intercept_[0]
xx = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
yy = -(w[0] * xx + b) / w[1]

plt.plot(xx, yy, 'k-')                  # Hyperplane
plt.plot(xx, yy + 1/w[1], 'k--')        # Margins
plt.plot(xx, yy - 1/w[1], 'k--')

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()