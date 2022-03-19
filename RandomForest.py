# Importing required packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from google.colab import drive
drive.mount('/content/drive')

# Read a csv_file
heart = pd.read_csv('/content/drive/MyDrive/AIProjects/saheart_1 withheader.csv')
heart.head()

# y is the target value
y = heart.iloc[:,0]

# X is the age of patients
X = heart.iloc[:,1:10]

# Learning
clf = RandomForestClassifier()
clf.fit(X,y)

prediction = clf.predict(X)

# probability of cases 1 and -1 for first fifth cases
pred_proba = clf.predict_proba(X)
print(pred_proba[:5,0]) # Probability of negative case
print(pred_proba[:5,1]) # Probability of positive case
