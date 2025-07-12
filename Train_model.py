import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer # For handling missing values
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib # For saving the model
import os
import sys # For more controlled exit

# --- Configuration & Setup ---
SCRIPT_NAME = "Train_model.py"
print(f"--- [{SCRIPT_NAME}] Starting Model Training Script ---")

# Define relative paths for dataset and model
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets the directory where the script is located
DATASET_FILENAME = 'kidney_disease.csv'
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', DATASET_FILENAME)

MODEL_DIR_NAME = 'model'
MODEL_FILENAME = 'ckd_model.pkl'
MODEL_DIR_PATH = os.path.join(BASE_DIR, MODEL_DIR_NAME)
MODEL_PATH = os.path.join(MODEL_DIR_PATH, MODEL_FILENAME)

# Ensure the 'model' directory exists, create if not
if not os.path.exists(MODEL_DIR_PATH):
    try:
        os.makedirs(MODEL_DIR_PATH)
        print(f"[{SCRIPT_NAME}] INFO: Model directory created at: {MODEL_DIR_PATH}")
    except OSError as e:
        print(f"[{SCRIPT_NAME}] ERROR: Could not create model directory at {MODEL_DIR_PATH}. Error: {e}")
        sys.exit(f"[{SCRIPT_NAME}] CRITICAL: Failed to create model directory. Exiting.")

# --- 1. Load Dataset ---
print(f"\n[{SCRIPT_NAME}] STEP 1: Loading dataset from '{DATASET_PATH}'...")
if not os.path.exists(DATASET_PATH):
    print(f"[{SCRIPT_NAME}] ERROR: Dataset file not found at '{DATASET_PATH}'.")
    print(f"[{SCRIPT_NAME}] Please ensure '{DATASET_FILENAME}' is in the 'dataset' subfolder of your project directory.")
    sys.exit(f"[{SCRIPT_NAME}] CRITICAL: Dataset not found. Exiting.")

try:
    df = pd.read_csv(DATASET_PATH)
    print(f"[{SCRIPT_NAME}] INFO: Dataset loaded successfully. Initial shape: {df.shape}")
    print(f"[{SCRIPT_NAME}] DEBUG: Initial columns: {df.columns.tolist()}")
except Exception as e:
    print(f"[{SCRIPT_NAME}] ERROR: Failed to load dataset. Error: {e}")
    sys.exit(f"[{SCRIPT_NAME}] CRITICAL: Dataset loading failed. Exiting.")

# --- 2. Initial Data Cleaning & Preprocessing ---
print(f"\n[{SCRIPT_NAME}] STEP 2: Performing initial data cleaning and preprocessing...")

# Replace common missing value placeholders with NaN
missing_value_placeholders = ['?', '\t?', ' ?', '', ' '] # Added empty string and space
df.replace(missing_value_placeholders, np.nan, inplace=True)
print(f"[{SCRIPT_NAME}] INFO: Replaced common missing value placeholders with NaN.")

# Drop 'id' column if it exists (usually not relevant for prediction)
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)
    print(f"[{SCRIPT_NAME}] INFO: 'id' column dropped.")

# Standardize column names (critical for consistency)
# This map covers many common variations found in CKD datasets.
# Adjust keys if your CSV has different original names.
column_rename_map = {
    'age': 'age', 'bp': 'blood_pressure', 'sg': 'specific_gravity', 'al': 'albumin',
    'su': 'sugar', 'rbc': 'red_blood_cells', 'pc': 'pus_cell',
    'pcc': 'pus_cell_clumps', 'ba': 'bacteria', 'bgr': 'blood_glucose_random',
    'bu': 'blood_urea', 'sc': 'serum_creatinine', 'sod': 'sodium',
    'pot': 'potassium', 'hemo': 'hemoglobin', 'pcv': 'packed_cell_volume',
    'wc': 'white_blood_cell_count', 'rc': 'red_blood_cell_count',
    'htn': 'hypertension', 'dm': 'diabetes_mellitus', 'cad': 'coronary_artery_disease',
    'appet': 'appetite', 'pe': 'pedal_edema', 'ane': 'anemia',
    'classification': 'class' # Target variable
}
# Only rename columns that exist in the DataFrame to avoid KeyErrors
actual_renames = {k: v for k, v in column_rename_map.items() if k in df.columns}
df.rename(columns=actual_renames, inplace=True)
print(f"[{SCRIPT_NAME}] INFO: Column names standardized. Renamed: {list(actual_renames.keys())}")
print(f"[{SCRIPT_NAME}] DEBUG: Columns after renaming: {df.columns.tolist()}")


# Define feature types for processing
# These are the *target names* after renaming
numeric_features_to_process = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
    'potassium', 'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count',
    'red_blood_cell_count'
]
categorical_features_to_process = [
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 'hypertension',
    'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'pedal_edema', 'anemia'
]

# Convert designated numeric columns to numeric type, coercing errors
print(f"[{SCRIPT_NAME}] INFO: Converting potential numeric columns to numeric type...")
for col in numeric_features_to_process:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"[{SCRIPT_NAME}] DEBUG: Converted '{col}' to numeric. NaN count: {df[col].isnull().sum()}")
    else:
        print(f"[{SCRIPT_NAME}] WARNING: Numeric feature '{col}' not found in DataFrame columns for type conversion.")

# --- 3. Imputation of Missing Values ---
print(f"\n[{SCRIPT_NAME}] STEP 3: Imputing missing values...")

# Impute numerical features with median
num_imputer = SimpleImputer(strategy='median')
for col in numeric_features_to_process:
    if col in df.columns:
        if df[col].isnull().sum() > 0: # Only impute if there are NaNs
            df[col] = num_imputer.fit_transform(df[[col]])
            print(f"[{SCRIPT_NAME}] INFO: Imputed missing values in numeric column '{col}' with median.")
        else:
            print(f"[{SCRIPT_NAME}] INFO: No missing values to impute in numeric column '{col}'.")

# Impute categorical features with mode (most frequent)
cat_imputer = SimpleImputer(strategy='most_frequent')
for col in categorical_features_to_process:
    if col in df.columns:
        if df[col].isnull().sum() > 0: # Only impute if there are NaNs
            # Store the original index and column name
            original_index = df.index
            imputed_values = cat_imputer.fit_transform(df[[col]])
            # After imputation, imputed_values is a NumPy array. 
            # We need to make sure it's flat (1D) before assigning back.
            # Then convert it to a pandas Series with the original index to ensure proper alignment.
            df[col] = pd.Series(imputed_values.flatten(), index=original_index)
            print(f"[{SCRIPT_NAME}] INFO: Imputed missing values in categorical column '{col}' with mode.")
        else:
            print(f"[{SCRIPT_NAME}] INFO: No missing values to impute in categorical column '{col}'.")

# --- 4. Mapping Categorical Features to Numeric ---
print(f"\n[{SCRIPT_NAME}] STEP 4: Mapping categorical features to numeric representations...")

# Specific mappings for known categorical features
# Ensure these mappings cover all expected values in your dataset after imputation
if 'red_blood_cells' in df.columns:
    df['red_blood_cells'] = df['red_blood_cells'].map({'normal': 1.0, 'abnormal': 0.0}).fillna(0.0) # Default to 0 if mapping fails
    print(f"[{SCRIPT_NAME}] INFO: Mapped 'red_blood_cells'. Unique values: {df['red_blood_cells'].unique()}")
if 'pus_cell' in df.columns:
    df['pus_cell'] = df['pus_cell'].map({'normal': 1.0, 'abnormal': 0.0}).fillna(0.0)
    print(f"[{SCRIPT_NAME}] INFO: Mapped 'pus_cell'. Unique values: {df['pus_cell'].unique()}")

# General mapping for binary 'yes'/'no' type features and the target class
# Ensure all original text values are lowercase and stripped of whitespace before mapping.
binary_text_map = {
    'yes': 1.0, 'no': 0.0,
    'good': 1.0, 'poor': 0.0,
    'present': 1.0, 'notpresent': 0.0,
    'ckd': 1.0, 'notckd': 0.0
}
cols_for_binary_map = [
    'pus_cell_clumps', 'bacteria', 'hypertension', 'diabetes_mellitus',
    'coronary_artery_disease', 'pedal_edema', 'anemia', 'appetite', 'class' # Target variable
]

for col in cols_for_binary_map:
    if col in df.columns:
        # Convert to string, strip whitespace, convert to lowercase for consistent mapping
        df[col] = df[col].astype(str).str.strip().str.lower()
        # Apply the map
        df[col] = df[col].replace(binary_text_map)
        # Convert to numeric, coercing any unmapped values to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # If NaNs remain (unmapped values), fill with mode (0 or 1, depending on what's more frequent after mapping)
        if df[col].isnull().sum() > 0:
            fill_value = df[col].mode()[0] if not df[col].mode().empty else 0.0 # Default to 0.0 if mode is empty
            df[col].fillna(fill_value, inplace=True)
            print(f"[{SCRIPT_NAME}] WARNING: Column '{col}' had unmapped values after binary mapping. Filled NaNs with mode ({fill_value}).")
        df[col] = df[col].astype(float) # Ensure it's float for consistency
        print(f"[{SCRIPT_NAME}] INFO: Mapped '{col}'. Unique values: {df[col].unique()}")
    else:
        print(f"[{SCRIPT_NAME}] WARNING: Categorical feature '{col}' for binary mapping not found in DataFrame columns.")


# --- 5. Feature Selection & Target Variable Preparation ---
print(f"\n[{SCRIPT_NAME}] STEP 5: Selecting features and preparing target variable...")

# Define the final list of features your model will use
# These names MUST match the column names in 'df' *after* all renaming and processing
final_model_features = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'serum_creatinine',
    'hemoglobin', 'packed_cell_volume', 'hypertension', 'diabetes_mellitus'
]

# Check if all selected model features are present and numeric
missing_model_features = []
non_numeric_model_features = []
for feature in final_model_features:
    if feature not in df.columns:
        missing_model_features.append(feature)
    elif not pd.api.types.is_numeric_dtype(df[feature]):
        non_numeric_model_features.append(feature)

if missing_model_features:
    print(f"[{SCRIPT_NAME}] ERROR: The following features required for the model are MISSING from the DataFrame: {missing_model_features}")
    print(f"[{SCRIPT_NAME}] Available columns: {df.columns.tolist()}")
    sys.exit(f"[{SCRIPT_NAME}] CRITICAL: Essential model features missing. Exiting.")
if non_numeric_model_features:
    print(f"[{SCRIPT_NAME}] ERROR: The following model features are NOT NUMERIC after processing: {non_numeric_model_features}")
    for feature in non_numeric_model_features:
        print(f"[{SCRIPT_NAME}] DEBUG: Dtype of '{feature}': {df[feature].dtype}, Unique values: {df[feature].unique()[:5]}")
    sys.exit(f"[{SCRIPT_NAME}] CRITICAL: Model features not numeric. Exiting.")

X = df[final_model_features]
print(f"[{SCRIPT_NAME}] INFO: Features (X) selected. Shape: {X.shape}")
print(f"[{SCRIPT_NAME}] DEBUG: X dtypes:\n{X.dtypes}")

# Target variable preparation
if 'class' not in df.columns:
    print(f"[{SCRIPT_NAME}] ERROR: Target variable 'class' not found in DataFrame.")
    sys.exit(f"[{SCRIPT_NAME}] CRITICAL: Target variable missing. Exiting.")

y = df['class']
# Ensure target is binary (0 or 1) and integer type
if not y.isin([0.0, 1.0]).all():
    print(f"[{SCRIPT_NAME}] ERROR: Target variable 'class' contains unexpected values after mapping: {y.unique()}")
    print(f"[{SCRIPT_NAME}] It should only contain 0.0 and 1.0. Please check mapping logic for 'class'.")
    sys.exit(f"[{SCRIPT_NAME}] CRITICAL: Target variable not binary. Exiting.")

y = y.astype(int) # Convert to integer type
print(f"[{SCRIPT_NAME}] INFO: Target variable (y) prepared. Shape: {y.shape}, Unique values: {y.unique()}")

# Final check for NaNs in X and y before splitting
if X.isnull().sum().sum() > 0:
    print(f"[{SCRIPT_NAME}] WARNING: NaNs detected in features (X) before splitting:\n{X.isnull().sum()[X.isnull().sum() > 0]}")
    print(f"[{SCRIPT_NAME}] This should ideally be zero after imputation. Review imputation steps.")
    # As a fallback, impute again, but this indicates an issue earlier
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True) # Or mode for categoricals if any were left
    print(f"[{SCRIPT_NAME}] INFO: Fallback imputation applied to X.")

if y.isnull().sum() > 0:
    print(f"[{SCRIPT_NAME}] WARNING: NaNs detected in target (y) before splitting: {y.isnull().sum()}")
    y.fillna(y.mode()[0], inplace=True) # Fill with mode
    print(f"[{SCRIPT_NAME}] INFO: Fallback imputation applied to y.")


# --- 6. Train/Test Split ---
print(f"\n[{SCRIPT_NAME}] STEP 6: Splitting data into training and testing sets...")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,     # 20% for testing
        random_state=42,   # For reproducibility
        stratify=y if y.nunique() > 1 else None # Stratify if more than one class in y
    )
    print(f"[{SCRIPT_NAME}] INFO: Data split successfully.")
    print(f"[{SCRIPT_NAME}] INFO: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"[{SCRIPT_NAME}] INFO: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
except ValueError as e:
    print(f"[{SCRIPT_NAME}] ERROR: Failed to split data. Error: {e}")
    print(f"[{SCRIPT_NAME}] DEBUG: y unique values before split: {y.unique()}, y length: {len(y)}")
    sys.exit(f"[{SCRIPT_NAME}] CRITICAL: Data splitting failed. Exiting.")


# --- 7. Train Random Forest Model ---
print(f"\n[{SCRIPT_NAME}] STEP 7: Training Random Forest Classifier model...")
try:
    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees in the forest
        random_state=42,       # For reproducibility
        class_weight='balanced'# Adjusts weights inversely proportional to class frequencies
    )
    model.fit(X_train, y_train)
    print(f"[{SCRIPT_NAME}] INFO: Model training completed successfully.")
except Exception as e:
    print(f"[{SCRIPT_NAME}] ERROR: Failed to train model. Error: {e}")
    sys.exit(f"[{SCRIPT_NAME}] CRITICAL: Model training failed. Exiting.")


# --- 8. Evaluate Model ---
print(f"\n[{SCRIPT_NAME}] STEP 8: Evaluating model performance on the test set...")
try:
    y_pred_test = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, zero_division=0)
    recall = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test)

    print(f"[{SCRIPT_NAME}] --- Model Evaluation Metrics ---")
    print(f"[{SCRIPT_NAME}] Accuracy:  {accuracy:.4f}")
    print(f"[{SCRIPT_NAME}] Precision: {precision:.4f} (Class 1)")
    print(f"[{SCRIPT_NAME}] Recall:    {recall:.4f} (Class 1)")
    print(f"[{SCRIPT_NAME}] F1-score:  {f1:.4f} (Class 1)")
    print(f"[{SCRIPT_NAME}] Confusion Matrix:\n{cm}")
    print(f"[{SCRIPT_NAME}] ------------------------------")
except Exception as e:
    print(f"[{SCRIPT_NAME}] ERROR: Failed during model evaluation. Error: {e}")
    # Continue to model saving if evaluation fails, but log the error.


# --- 9. Save the Trained Model ---
print(f"\n[{SCRIPT_NAME}] STEP 9: Saving the trained model to '{MODEL_PATH}'...")
try:
    joblib.dump(model, MODEL_PATH)
    print(f"[{SCRIPT_NAME}] INFO: Model saved successfully to {MODEL_PATH}")
except Exception as e:
    print(f"[{SCRIPT_NAME}] ERROR: Failed to save the model. Error: {e}")
    sys.exit(f"[{SCRIPT_NAME}] CRITICAL: Model saving failed. Exiting.")

print(f"\n--- [{SCRIPT_NAME}] Model Training Script Completed Successfully ---")