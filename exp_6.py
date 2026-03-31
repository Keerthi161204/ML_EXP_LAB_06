import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------
# LOAD DATASET
# ---------------------------
data = load_breast_cancer()
X = data.data
y = data.target

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------------------------
# DECISION TREE + CV
# ---------------------------
dt = DecisionTreeClassifier(max_depth=5, criterion='gini')

dt_scores = cross_val_score(dt, X, y, cv=5)
print("Decision Tree CV Accuracy:", dt_scores)
print("Decision Tree Avg CV Accuracy:", dt_scores.mean())

dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\nDecision Tree Test Accuracy:", accuracy_score(y_test, dt_pred))

# ---------------------------
# RANDOM FOREST + CV
# ---------------------------
rf = RandomForestClassifier(n_estimators=100, max_depth=5)

rf_scores = cross_val_score(rf, X, y, cv=5)
print("\nRandom Forest CV Accuracy:", rf_scores)
print("Random Forest Avg CV Accuracy:", rf_scores.mean())

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest Test Accuracy:", accuracy_score(y_test, rf_pred))

# ---------------------------
# REPORT
# ---------------------------
print("\nRandom Forest Classification Report:\n")
print(classification_report(y_test, rf_pred))
