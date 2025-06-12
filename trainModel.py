import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv('coords.csv',
                 on_bad_lines='skip',  # pandas ≥1.3: silently drop malformed rows
                 engine='python'  # required for on_bad_lines
                 )
df = df.dropna()
X = df.drop(columns=['class'])
y = df['class']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pipeline
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
model = pipeline.fit(X_train, y_train)
print('Train accuracy:', model.score(X_train, y_train))
print('Test accuracy:', model.score(X_test, y_test))
# Save model
with open('body_language_model.pkl', 'wb') as f:
    pickle.dump(model, f)
