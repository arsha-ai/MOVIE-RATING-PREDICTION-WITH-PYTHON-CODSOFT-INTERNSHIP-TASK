import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    data_path = r"C:\Users\siddh\OneDrive\Desktop\codsoft intern\MOVIE RATING PREDICTION WITH PYTHON\IMDb Movies India.csv"
    
    try:
        data = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(data_path, encoding='iso-8859-1')
        except UnicodeDecodeError:
            data = pd.read_csv(data_path, encoding='cp1252')
    
    # Print column names and first few rows for debugging
    print("Columns in the dataset:", data.columns)
    print("\nFirst few rows of the dataset:")
    print(data.head())

    # Check if 'Year' column exists
    if 'Year' in data.columns:
        # Convert 'Year' to numeric, handling potential string values
        data['Year'] = pd.to_numeric(data['Year'].astype(str).str.extract('(\d+)', expand=False), errors='coerce')
    else:
        print("'Year' column not found in the dataset.")

    # Check if 'Duration' column exists
    if 'Duration' in data.columns:
        # Convert 'Duration' to numeric, handling potential string values
        data['Duration'] = pd.to_numeric(data['Duration'].astype(str).str.extract('(\d+)', expand=False), errors='coerce')
    else:
        print("'Duration' column not found in the dataset.")

    # Check if 'Votes' column exists
    if 'Votes' in data.columns:
        # Convert 'Votes' to numeric
        data['Votes'] = pd.to_numeric(data['Votes'], errors='coerce')
    else:
        print("'Votes' column not found in the dataset.")

    return data

def create_model_pipeline():
    numeric_features = ['Year', 'Duration', 'Votes']
    categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    return model

def train_and_evaluate_model(data, model):
    features = ['Year', 'Duration', 'Votes', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    target = 'Rating'
    
    data_clean = data.dropna(subset=features + [target])
    
    X = data_clean[features]
    y = data_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-validation R2 scores: {cv_scores}")
    print(f"Mean CV R2 score: {np.mean(cv_scores):.4f}")
    
    return model

def get_user_input():
    print("\nEnter Movie Details:")
    year = int(input("Year (1900-2023): "))
    duration = int(input("Duration (minutes, 1-300): "))
    votes = int(input("Number of Votes (1-1000000): "))
    genre = input("Genre: ")
    director = input("Director: ")
    actor1 = input("Actor 1: ")
    actor2 = input("Actor 2: ")
    actor3 = input("Actor 3: ")
    
    return pd.DataFrame({
        'Year': [year],
        'Duration': [duration],
        'Votes': [votes],
        'Genre': [genre],
        'Director': [director],
        'Actor 1': [actor1],
        'Actor 2': [actor2],
        'Actor 3': [actor3]
    })

def predict_and_explain(model, input_data):
    prediction = model.predict(input_data)
    
    print(f"\nPrediction Result:")
    print(f"The predicted rating for this movie is: {prediction[0]:.2f}")
    
    feature_importance = model.named_steps['regressor'].feature_importances_
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.show()

def main():
    print("Movie Rating Prediction")
    
    data = load_and_preprocess_data()
    model = create_model_pipeline()
    trained_model = train_and_evaluate_model(data, model)
    
    while True:
        input_data = get_user_input()
        predict_and_explain(trained_model, input_data)
        
        another = input("\nDo you want to predict another movie? (yes/no): ").lower()
        if another != 'yes':
            break

    print("Thank you for using the Movie Rating Prediction model!")

if __name__ == "__main__":
    main()