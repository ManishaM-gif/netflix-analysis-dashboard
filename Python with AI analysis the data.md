# netflix-analysis-dashboard
"Interactive Netflix content analysis dashboard built in Power BI using dataset of 9,000+ titles."
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# STEP 1: Load Data
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/netflix_titles.csv')

# Keep all data
print("Initial shape:", df.shape)
print(df.info())
print(df.isnull().sum())

# STEP 2: Handle missing values (but DO NOT drop rows unless critical)
# Fill missing text fields with "Unknown"
text_cols = ['director', 'cast', 'rating', 'country']
for col in text_cols:
    df[col] = df[col].fillna('Unknown')

# Fill missing date_added for safety
df['date_added'] = df['date_added'].fillna(method='ffill')

# Drop duplicates if any
df.drop_duplicates(inplace=True)

# STEP 3: Feature Engineering

## 3.1 Duration - Numeric format
def parse_duration(row):
    if pd.isnull(row):
        return np.nan
    tokens = row.split()
    try:
        return int(tokens[0])
    except:
        return np.nan

df['duration_int'] = df['duration'].apply(parse_duration)
df['duration_unit'] = df['duration'].str.extract(r'([a-zA-Z]+)', expand=False)

# Fill missing duration with median by type
df['duration_int'] = df.groupby('type')['duration_int'].transform(lambda x: x.fillna(x.median()))
df['duration_unit'] = df['duration_unit'].fillna('min')

## 3.2 Country - keep Top 5
top_countries = df['country'].value_counts().nlargest(5).index
df['country_top5'] = df['country'].apply(lambda x: x if x in top_countries else 'Other')

## 3.3 Date fields
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['added_year'] = df['date_added'].dt.year
df['added_month'] = df['date_added'].dt.month

## 3.4 Encode categorical variables
le_rating = LabelEncoder()
df['rating_encoded'] = le_rating.fit_transform(df['rating'])

# One-hot for country_top5
df = pd.get_dummies(df, columns=['country_top5'], prefix='country')

# STEP 4: Prepare data for ML

# Select features
features = ['duration_int', 'added_year', 'added_month', 'rating_encoded'] + [col for col in df.columns if col.startswith('country_')]
X = df[features]
y = df['type'].map({'Movie': 0, 'TV Show': 1})  # Binary: 0 = Movie, 1 = TV Show

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 5: Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# Evaluation
print("Accuracy:", rf.score(X_test, y_test))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind='bar', figsize=(10, 4), title='Feature Importance')
plt.tight_layout()
plt.show()

# STEP 6: Export prediction results for Power BI
df['prediction_proba_TV_Show'] = rf.predict_proba(X)[:, 1]
df['prediction_label'] = rf.predict(X)

# Export
df.to_csv('/content/drive/MyDrive/Colab Notebooks/netflix_predictions_output.csv', index=False)
print("Exported to Power BI!")
