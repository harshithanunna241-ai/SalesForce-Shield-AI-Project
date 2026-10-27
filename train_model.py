
from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

train_path = DATA_DIR / "prepared" / "training_clean.csv"
test_path = DATA_DIR / "prepared" / "testing_clean.csv"

print("Loading prepared datasets...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Binary target: 0 = normal, 1 = attack
target = "label"

# Remove target and identifiers that should not be used as predictors
drop_columns = [target, "attack_cat", "id"]

X_train = train_df.drop(
    columns=[c for c in drop_columns if c in train_df.columns]
)
y_train = train_df[target].astype(int)

X_test = test_df.drop(
    columns=[c for c in drop_columns if c in test_df.columns]
)
y_test = test_df[target].astype(int)

# Identify numeric and categorical features
numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Convert numeric columns safely and impute missing values
for col in numeric_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=2
    ))
])

print("Training model...")
model.fit(X_train, y_train)

print("Evaluating model...")
predictions = model.predict(X_test)

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

model_path = BASE_DIR / "shieldai_model.joblib"
joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")
