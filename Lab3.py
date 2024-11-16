import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("titanic.csv")
df = df.drop(columns=["PassengerId", "Name", "Ticket"])

le = preprocessing.LabelEncoder()
survived_encoded = le.fit_transform(df['Survived'])
pclass_encoded = le.fit_transform(df['Pclass'])
sex_encoded = le.fit_transform(df['Sex'])
age_encoded = le.fit_transform(df['Age'].fillna(df['Age'].mean()))
sibsp_encoded = le.fit_transform(df['SibSp'])
parch_encoded = le.fit_transform(df['Parch'])
fare_encoded = le.fit_transform(df['Fare'])
cabin_encoded = le.fit_transform(df['Cabin'].fillna('U'))
embarked_encoded = le.fit_transform(df['Embarked'].fillna('S'))

features = list(zip(pclass_encoded, sex_encoded, age_encoded, sibsp_encoded, parch_encoded, fare_encoded, cabin_encoded, embarked_encoded))

features_train, features_test, label_train, label_test = train_test_split(features, survived_encoded, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(features_train, label_train)

predicted = model.predict(features_test)

print("Prediction:", predicted)

conf_mat = confusion_matrix(label_test, predicted)
print("Confusion Matrix:")
print(conf_mat)

accuracy = accuracy_score(label_test, predicted)
print("Accuracy:", accuracy)
