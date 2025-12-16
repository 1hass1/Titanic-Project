import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
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
print('\n=== Random Forest Results: ===\n')
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_RF)}")

# prints out Accuracy score, Precision score, recall score, f1 score, macro avg, weighted avg (all together)
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_RF)}")



# Predict and evaluate Logistic Regression
y_pred_LogReg = Log_Reg_pipline.predict(X_test)
print("\n=== Logistic Regression Results: ===\n")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_LogReg)}")

# prints out Accuracy score, Precision score, recall score, f1 score, macro avg, weighted avg (all together)
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_LogReg)}")


# Feature importance in Random Forest
preprocessor = RF_pipeline.named_steps['preprocessor']
print("\n=== Random Forest Feature Results: ===\n")
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


# Cross-validation for Random Forest:
# compares different models' performance vs. true predictions
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}


cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cross_val_rf = cross_validate(
    RF_pipeline,
    X,
    y,
    cv=cross_val,
    scoring=scoring,
    n_jobs=-1
)

print("\n\n=== Random Forest Cross-Validation: ===")
for metric in scoring:
    mean_score = cross_val_rf[f"test_{metric}"].mean()
    std_score = cross_val_rf[f"test_{metric}"].std()
    print(f"{metric}: {mean_score:.3f} ± {std_score:.3f}")


# cross validation for logistic regression
cross_val_lr = cross_validate(
    Log_Reg_pipline,
    X,
    y,
    cv=cross_val,
    scoring=scoring,
    n_jobs=-1
)   

print("\n=== Logistic Regression Cross-Validation: ===")
for metric in scoring:
    mean_score = cross_val_lr[f"test_{metric}"].mean()
    std_score = cross_val_lr[f"test_{metric}"].std()
    print(f"{metric}: {mean_score:.3f} ± {std_score:.3f}")



# Finetuning hyperparameters using GridSearchCV
# Random Forest Model
# Evaluate Finetuned model

rf_param_grid = {
    "classifier model__n_estimators": [100, 200],
    "classifier model__max_depth": [None, 5, 10],
    "classifier model__min_samples_split": [2, 5],
    "classifier model__min_samples_leaf": [1, 2]
}

rf_grid = GridSearchCV(
    estimator=RF_pipeline,
    param_grid=rf_param_grid,
    scoring="f1",
    cv=cross_val,
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)

print("\n=== Random Forest GridSearchCV Results: ===")
print("Best parameters:", rf_grid.best_params_)
print("Best cross-validation score:", rf_grid.best_score_)


best_rf = rf_grid.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

print("\n=== Finetuned Random Forest Results: ===")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_best_rf)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_best_rf)}")
