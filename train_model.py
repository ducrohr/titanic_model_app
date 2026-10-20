
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import joblib

# STEP 1: Load the checked-in Titanic dataset
df = pd.read_csv(Path(__file__).parent / 'data' / 'titanic.csv')

# STEP 2: Select only the columns we want to use for training
df = df[['survived', 'pclass', 'sex', 'age']].dropna()  # Keep only useful columns & remove rows with missing data

# STEP 3: Convert text into numbers (model only understands numbers)
df['sex'] = df['sex'].map({'male': 0, 'female': 1})  # male = 0, female = 1

# STEP 4: Split the data into features (X) and target/output (y)
X = df[['pclass', 'sex', 'age']].to_numpy()  # Deeploy sends positional values
y = df['survived']                      # y = what we're trying to predict (survived or not)

# STEP 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)  # 75% train, 25% test

# STEP 6: Create the model
model = LogisticRegression(max_iter=1000)  # This is our logistic regression model

# STEP 7: Train the model using the training data
model.fit(X_train, y_train)             # The model learns from data

# STEP 8: Save the model to a file so we can use it later in app.py
joblib.dump(model, Path(__file__).parent / 'model' / 'model.joblib')
