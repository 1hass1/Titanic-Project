import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from  sklearn.compose import ColumnTransformer

Titanic_Data = pd.read_csv(r"c:\Users\rayan\Desktop\AI Projects\Titanic Project\Titanic_Data.csv")
# print(Titanic_Data.head())

Titanic_Data['Age'].fillna(Titanic_Data['Age'].median(), inplace=True)
Titanic_Data['Embarked'].fillna(Titanic_Data['Embarked'].mode()[0], inplace=True)



X = Titanic_Data.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
y = Titanic_Data["Survived"]

preprocessor = ColumnTransformer('num', OneHotEncoder(), ['Sex', 'Embarked'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)
y_pred = forest_clf.predict(X_test)
forest_clf.score(X_test, y_test)