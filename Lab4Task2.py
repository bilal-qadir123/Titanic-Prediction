import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
df = pd.read_csv("car_data.csv")
le = preprocessing.LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
features = df[["User ID", "Gender", "Age", "AnnualSalary"]].values
labels = df["Purchased"].values
features_train, features_test, label_train, label_test = train_test_split(features, labels, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(features_train, label_train)
sample_data = [[123, le.transform(["Male"])[0], 30, 50000]]  # User ID, Gender, Age, Annual Salary
sample_prediction = model.predict(sample_data)
print("Prediction for sample data:", "Purchased" if sample_prediction[0] == 1 else "Not Purchased") 
