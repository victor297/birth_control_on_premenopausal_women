import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor as LinearRegression_
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputRegressor as MultiLinearRegression 
from sklearn.linear_model import LinearRegression

# Load the dataset
try:
    df = pd.read_csv('dataset_with_age_at_test.csv')
except FileNotFoundError:
    st.error('Dataset not found. Please check the file path.')
    raise

# Define X and y
X = df.drop(columns=['Wife\'s age', 'Age when drug was used', 'Effect of the drug'])  # Exclude Wife's age
y = df[['Age when drug was used', 'Effect of the drug']]

# Categorical and numerical features
categorical_features = ['Wife\'s education', 'Wife\'s religion', 'Wife\'s now working?', 'Husband\'s occupation',
                        'Standard-of-living index', 'Media exposure', 'Contraceptive method used']
numeric_features = ['Number of children ever born', 'Age at test']  # Include Age at test

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiLinearRegression(LinearRegression_(random_state=42)))
])

# Fit the model
model.fit(X, y)

# Define the mapping ranges
effect_mapping = {
    "Irregular bleeding or spotting": (1.0, 1.5),
    "Potential bone density loss": (1.5, 2.0),
    "Weight gain and mood changes": (2.0, 2.5),
    "Increased risk of blood clots": (2.5, 3.0),
    "Slightly increased risk of breast cancer": (3.0, 3.5),
    "Possible cardiovascular issues": (3.5, 4.0)
}

# Function to map the predicted effect value to a category
def map_effect(predicted_value):
    for effect, (low, high) in effect_mapping.items():
        if low <= predicted_value < high:
            return effect
    return "Unknown effect"

# Streamlit app
st.subheader('Detecting the use of Birth Control Drug in premenopausal woman using LinearRegression')
st.write('By Oladipo Grace Tolulope')
st.write('21d/47cs/01566')

st.sidebar.header('User Input Parameters')

# Function to get user inputs
def get_user_inputs():
    user_inputs = {}
    user_inputs['Wife\'s education'] = st.sidebar.selectbox('Wife\'s education', sorted(df['Wife\'s education'].unique()))
    user_inputs['Husband\'s education'] = st.sidebar.selectbox('Husband\'s education', sorted(df['Husband\'s education'].unique()))
    user_inputs['Number of children ever born'] = st.sidebar.slider('Number of children ever born', int(X['Number of children ever born'].min()), int(X['Number of children ever born'].max()), int(X['Number of children ever born'].mean()))
    user_inputs['Wife\'s religion'] = st.sidebar.selectbox('Wife\'s religion', sorted(df['Wife\'s religion'].unique()))
    user_inputs['Wife\'s now working?'] = st.sidebar.selectbox('Wife\'s now working?', sorted(df['Wife\'s now working?'].unique()))
    user_inputs['Husband\'s occupation'] = st.sidebar.selectbox('Husband\'s occupation', sorted(df['Husband\'s occupation'].unique()))
    user_inputs['Standard-of-living index'] = st.sidebar.selectbox('Standard-of-living index', sorted(df['Standard-of-living index'].unique()))
    user_inputs['Media exposure'] = st.sidebar.selectbox('Media exposure', sorted(df['Media exposure'].unique()))
    user_inputs['Contraceptive method used'] = st.sidebar.selectbox('Contraceptive method used', sorted(df['Contraceptive method used'].unique()))
    user_inputs['Age at test'] = st.sidebar.slider('Age at test', int(X['Age at test'].min()), int(X['Age at test'].max()), int(X['Age at test'].mean()))
    return pd.DataFrame([user_inputs])

# Get user inputs
user_inputs_df = get_user_inputs()

# Display user inputs
st.subheader('User Input:')
st.write(user_inputs_df)

# Make predictions
predictions = model.predict(user_inputs_df)

# Map the predicted effect to a category
predicted_effect_value = predictions[0][1]
effect_category = map_effect(predicted_effect_value)

# Display predictions
st.subheader('Predicted Results:')
st.write('Predicted Age when drug was used:', predictions[0][0])
st.write('Predicted Effect of the drug:', predicted_effect_value)
st.write('Categorized Effect of the drug:', effect_category)

# Additional information
st.sidebar.markdown('---')
st.sidebar.subheader('About')
st.sidebar.text('This app predicts the Age when drug was used and the Effect of the drug based on user inputs, including Age at test.')
