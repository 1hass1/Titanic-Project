import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline


Titanic_Data = pd.read_csv(r"c:\Users\rayan\Desktop\AI Projects\Titanic Project\Titanic_Data.csv")
# print(Titanic_Data.head())

Titanic_Data['Age'].fillna(Titanic_Data['Age'].median(), inplace=True)
Titanic_Data['Embarked'].fillna(Titanic_Data['Embarked'].mode()[0], inplace=True)

# Feature Engineering for "Name" Feature (Extracting "Title" of passengers for more accurate predictions)
Titanic_Data["Title"] = Titanic_Data["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
# normalize similar titles
Titanic_Data["Title"] = Titanic_Data["Title"].replace(['Lady', 'Countess','Capt', 'Col',
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
Titanic_Data["Title"] = Titanic_Data["Title"].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})


# choosing relative features (dropping "Name" after extracting "Title")
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title"]
num_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
X = Titanic_Data[features]
y = Titanic_Data["Survived"]


# Preprocessing (converting all str into numerical values)
str_features = ["Sex", "Embarked", "Title"]
preprocessor = ColumnTransformer(transformers=[('str', OneHotEncoder(handle_unknown='ignore'), str_features)], remainder='passthrough')
preprocessor = ColumnTransformer(
    transformers=[
        ('str', OneHotEncoder(handle_unknown='ignore'), str_features),
        ('num', 'passthrough', num_features)
    ]
)

# train-test split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Build Pipeline (Random Forest and Logistic Regression)
RF_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier model', RandomForestClassifier(n_estimators=100, random_state=42))])
Log_Reg_pipline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier model', LogisticRegression(max_iter=1000, random_state=42))])

# Train models
RF_pipeline.fit(X_train, y_train)
Log_Reg_pipline.fit(X_train, y_train)

# Predict and evaluate Random Forest
y_pred_RF = RF_pipeline.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred_RF))
print(confusion_matrix(y_test, y_pred_RF))
print(classification_report(y_test, y_pred_RF))

# Predict and evaluate Logistic Regression
y_pred_LogReg = Log_Reg_pipline.predict(X_test)
print("\nLogistic Regression Results:\n")
print('Accuracy:', accuracy_score(y_test, y_pred_LogReg))
print(confusion_matrix(y_test, y_pred_LogReg))
print(classification_report(y_test, y_pred_LogReg))

# Feature importance in Random Forest
preprocessor = RF_pipeline.named_steps['preprocessor']
print("\nRandom Forest Feature Results:\n")
str_encoder = preprocessor.named_transformers_['str']
str_feature_names = str_encoder.get_feature_names_out(str_features)
feature_names = list(str_feature_names) + num_features

RF_model = RF_pipeline.named_steps['classifier model']

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': RF_model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\n Top 10 Most Important Features in Random Forest:\n")
print(importance_df.head(10))