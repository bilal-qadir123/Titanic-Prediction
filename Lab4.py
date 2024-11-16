import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
df = pd.read_csv('adult.csv')
le = preprocessing.LabelEncoder()
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
X = df.drop('income', axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
sample_data = pd.DataFrame([[39, 2, 77516, 9, 13, 2, 6, 0, 0, 0, 15024, 0, 40, 39]], columns=X.columns)
prediction = model.predict(sample_data)
print("Predicted Income:", le.inverse_transform(prediction)) 