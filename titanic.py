import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load Dataset
url = "https://raw.githubusercontent.com"
df = pd.read_csv(url)

# 2. Preprocess Data
# Select target features: Age, Sex, Fare, Pclass (Class)
df = df[['Age', 'Sex', 'Fare', 'Pclass', 'Survived']]

# Handle missing Age values by filling with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Encode Sex: Male=0, Female=1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 3. Train Model
X = df[['Age', 'Sex', 'Fare', 'Pclass']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Evaluate Model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 5. Predict for specific passenger
# 30-year-old male, 3rd class, median 3rd-class fare (~$8.05)
passenger = [[30, 0, 8.05, 3]] 
prediction = model.predict(passenger)
print(f"\nPrediction for 30yo Male (3rd Class): {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")
