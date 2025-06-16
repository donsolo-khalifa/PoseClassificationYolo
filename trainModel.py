import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load data
df = pd.read_csv('coords.csv'
                 # on_bad_lines='skip',  # pandas ≥1.3: silently drop malformed rows
                 # engine='python'  # required for on_bad_lines
                 )
df = df.dropna()
X = df.drop('class', axis=1)
y = df['class']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# Pipeline
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), # Increased max_iter for convergence
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

trained_models = {}
for name, pipeline in pipelines.items():
    print(f"Training {name} model...")
    model = pipeline.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} model training complete.")

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  {name} Model Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    model_filename = f'body_language_{name}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved to {model_filename}\n")

print("All models trained and saved.")
