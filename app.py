import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor as MultiLinearRegression
from sklearn.ensemble import RandomForestRegressor as LinearRegression_
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel

# Step 1: Load and Visualize Data
df = pd.read_csv('dataset_with_age_at_test.csv')

# Data visualization
sns.pairplot(df)
plt.show()

# Check for outliers in numerical features
numeric_features = ['Wife\'s age', 'Number of children ever born', 'Stature']
plt.figure(figsize=(15, 6))
for i, feature in enumerate(numeric_features):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(y=df[feature])
    plt.title(f'Boxplot of {feature}')
plt.show()

# Step 2: Data Preprocessing
X = df.drop(columns=['Age when drug was used', 'Effect of the drug'])
y = df[['Age when drug was used', 'Effect of the drug']]

categorical_features = ['Wife\'s education', 'Husband\'s education', 'Wife\'s religion',
                        'Wife\'s now working?', 'Husband\'s occupation', 'Standard-of-living index',
                        'Media exposure', 'Contraceptive method used', 'Birth control drug type', 
                        'Race', 'Complexion']
numeric_features = ['Wife\'s age', 'Number of children ever born', 'Stature']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Selection and Training (Linear Regression)
model_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiLinearRegression(LinearRegression_(random_state=42)))
])

# Train and evaluate the initial model
model_rf.fit(X_train, y_train)
predictions_rf = model_rf.predict(X_test)
score_rf = r2_score(y_test, predictions_rf)
print(f"Initial Model (Linear Regression) R^2 score: {score_rf:.2f}")

# Step 4: Feature Importance
importances = model_rf.named_steps['regressor'].estimators_[0].feature_importances_
feature_names = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features).tolist() + numeric_features
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances.sort_values(by='Importance', ascending=False))
plt.title('Feature Importances')
plt.show()

# Step 5: Hyperparameter Tuning
param_grid_rf = {
    'regressor__estimator__n_estimators': [50, 100, 200],
    'regressor__estimator__max_depth': [None, 10, 20],
    'regressor__estimator__min_samples_split': [2, 5, 10],
    'regressor__estimator__min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(model_rf, param_grid_rf, cv=3, scoring='r2', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

best_model_rf = grid_search_rf.best_estimator_
predictions_rf_tuned = best_model_rf.predict(X_test)
score_rf_tuned = r2_score(y_test, predictions_rf_tuned)
print(f"Tuned Model (Linear Regression) R^2 score: {score_rf_tuned:.2f}")
print(f"Best parameters: {grid_search_rf.best_params_}")

# Step 6: Evaluate with Tuned Model
# Optional: Evaluate with Gradient Boosting Regressor
model_gb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiLinearRegression(GradientBoostingRegressor(random_state=42)))
])

model_gb.fit(X_train, y_train)
predictions_gb = model_gb.predict(X_test)
score_gb = r2_score(y_test, predictions_gb)
print(f"Model (Gradient Boosting Regressor) R^2 score: {score_gb:.2f}")

# Step 7: Feature Selection
selector = SelectFromModel(best_model_rf.named_steps['regressor'].estimators_[0], threshold='median')
selector.fit(X_train, y_train)  # Fit on the original X_train and y_train

# Get selected features and their names
support = selector.get_support()
feature_names_selected = X.columns[support].tolist()

# Transform X_train and X_test with selected features
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Convert X_train_selected and X_test_selected to DataFrame
X_train_selected = pd.DataFrame(X_train_selected, columns=feature_names_selected)
X_test_selected = pd.DataFrame(X_test_selected, columns=feature_names_selected)

# Use ColumnTransformer with pandas columns directly
preprocessor_selected = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), feature_names_selected)
    ])

# Append the feature selection step to the pipeline
model_selected = Pipeline(steps=[
    ('preprocessor', preprocessor_selected),
    ('regressor', MultiLinearRegression(LinearRegression_(random_state=42)))
])

# Fit and evaluate the model with selected features
model_selected.fit(X_train_selected, y_train)
predictions_selected = model_selected.predict(X_test_selected)
score_selected = r2_score(y_test, predictions_selected)
print(f"Model with selected features R^2 score: {score_selected:.2f}")
