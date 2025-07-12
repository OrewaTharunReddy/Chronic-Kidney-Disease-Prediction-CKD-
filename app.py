print("--- [app.py] Script execution started ---") # VERY FIRST LINE

import sys # Import sys for more info if needed

try:
    print("--- [app.py] Attempting to import Flask ---")
    from flask import Flask, render_template, request
    print("--- [app.py] Flask imported successfully ---")

    print("--- [app.py] Attempting to import pandas ---")
    import pandas as pd
    print("--- [app.py] pandas imported successfully ---")

    print("--- [app.py] Attempting to import numpy ---")
    import numpy as np
    print("--- [app.py] numpy imported successfully ---")

    print("--- [app.py] Attempting to import joblib ---")
    import joblib
    print("--- [app.py] joblib imported successfully ---")

    print("--- [app.py] Attempting to import os ---")
    import os
    print("--- [app.py] os imported successfully ---")

except ImportError as ie:
    print(f"--- [app.py] CRITICAL IMPORT ERROR: {ie} ---")
    sys.exit(f"Import error, cannot continue: {ie}") # Exit if an import fails
except Exception as e_import:
    print(f"--- [app.py] UNEXPECTED ERROR DURING IMPORTS: {e_import} ---")
    sys.exit(f"Unexpected error during imports: {e_import}")


print("--- [app.py] All imports seem successful. Initializing Flask app... ---")
try:
    app = Flask(__name__)
    print("--- [app.py] Flask app initialized. ---")
except Exception as e_flask_init:
    print(f"--- [app.py] ERROR INITIALIZING FLASK APP: {e_flask_init} ---")
    sys.exit(f"Error initializing Flask: {e_flask_init}")


print("--- [app.py] Defining MODEL_PATH... ---")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'ckd_model.pkl')
# Corrected MODEL_PATH to be absolute for robustness in this debugging phase
print(f"--- [app.py] MODEL_PATH defined as: {MODEL_PATH} ---")


print("--- [app.py] Attempting to load model... ---")
model = None # Initialize model to None
try:
    if not os.path.exists(MODEL_PATH):
        print(f"--- [app.py] ERROR: Model file does NOT exist at {MODEL_PATH} ---")
        # We will let it proceed to the 'if not model:' check in index()
    else:
        print(f"--- [app.py] Model file found at {MODEL_PATH}. Loading... ---")
        model = joblib.load(MODEL_PATH)
        print(f"--- [app.py] Model loaded successfully from {MODEL_PATH} ---")
except FileNotFoundError: # This specific exception is good to catch
    print(f"--- [app.py] EXCEPTION (FileNotFoundError): Model file not found at {MODEL_PATH}. ---")
    # model remains None
except Exception as e_model_load:
    print(f"--- [app.py] EXCEPTION during model load: {e_model_load} ---")
    # model remains None


print("--- [app.py] Defining FEATURES_INFO and ORDERED_FEATURES... ---")
FEATURES_INFO = {
    'age': {'label': 'Age (years)', 'type': 'number', 'step': '1'},
    'blood_pressure': {'label': 'Blood Pressure (mm/Hg)', 'type': 'number', 'step': '1'},
    'specific_gravity': {'label': 'Specific Gravity (e.g., 1.010)', 'type': 'number', 'step': '0.001'},
    'albumin': {'label': 'Albumin (0-5)', 'type': 'number', 'step': '1'},
    'serum_creatinine': {'label': 'Serum Creatinine (mg/dL)', 'type': 'number', 'step': '0.1'},
    'hemoglobin': {'label': 'Hemoglobin (gms/dL)', 'type': 'number', 'step': '0.1'},
    'packed_cell_volume': {'label': 'Packed Cell Volume (%)', 'type': 'number', 'step': '1'},
    'hypertension': {'label': 'Hypertension', 'type': 'select', 'options': {'No': 0.0, 'Yes': 1.0}},
    'diabetes_mellitus': {'label': 'Diabetes Mellitus', 'type': 'select', 'options': {'No': 0.0, 'Yes': 1.0}}
}
ORDERED_FEATURES = ['age', 'blood_pressure', 'specific_gravity', 'albumin',
                    'serum_creatinine', 'hemoglobin', 'packed_cell_volume',
                    'hypertension', 'diabetes_mellitus']
print("--- [app.py] FEATURES_INFO and ORDERED_FEATURES defined. ---")


print("--- [app.py] Defining Flask route / ... ---")
@app.route("/", methods=["GET", "POST"])
def index():
    print("--- [app.py] index() function called. ---")
    prediction_text = None
    error_message = None
    form_values = {feature: '' for feature in ORDERED_FEATURES}

    if not model:
        print("--- [app.py] Model is None in index(). Setting error message. ---")
        error_message = "Model not loaded or failed to load. Predictions unavailable. Please check server logs."
        return render_template("index.html", features_info=FEATURES_INFO,
                               ordered_features=ORDERED_FEATURES, error_message=error_message,
                               form_values=form_values)

    if request.method == "POST":
        print("--- [app.py] index() - POST request received. ---")
        input_data_dict = {}
        valid_input = True
        for feature_name in ORDERED_FEATURES:
            form_value = request.form.get(feature_name, '').strip()
            form_values[feature_name] = form_value
            feature_type = FEATURES_INFO[feature_name]['type']

            if not form_value:
                error_message = f"'{FEATURES_INFO[feature_name]['label']}' cannot be empty."
                valid_input = False; break

            if feature_type == 'number':
                try: input_data_dict[feature_name] = float(form_value)
                except ValueError:
                    error_message = f"Invalid numeric value for '{FEATURES_INFO[feature_name]['label']}'."; valid_input = False; break
            elif feature_type == 'select':
                try:
                    val = float(form_value)
                    if val not in FEATURES_INFO[feature_name]['options'].values():
                        error_message = f"Invalid option for '{FEATURES_INFO[feature_name]['label']}'."; valid_input = False; break
                    input_data_dict[feature_name] = val
                except ValueError:
                    error_message = f"Invalid selection for '{FEATURES_INFO[feature_name]['label']}'."; valid_input = False; break

        if valid_input:
            print("--- [app.py] index() - Input is valid. Attempting prediction. ---")
            try:
                ordered_input_values = [input_data_dict[fname] for fname in ORDERED_FEATURES]
                df_input = pd.DataFrame([ordered_input_values], columns=ORDERED_FEATURES)
                prediction_result = model.predict(df_input)[0]
                prediction_proba = model.predict_proba(df_input)[0]
                confidence = prediction_proba[1] if prediction_result == 1 else prediction_proba[0]
                status = "CKD Detected" if prediction_result == 1 else "No CKD Detected"
                prediction_text = f"{status} (Confidence: {confidence*100:.2f}%)"
                print(f"--- [app.py] index() - Prediction made: {prediction_text} ---")
            except Exception as e_predict:
                error_message = f"Prediction error: {str(e_predict)}"
                print(f"--- [app.py] index() - EXCEPTION during prediction: {e_predict} ---")
        else:
            print(f"--- [app.py] index() - Input was invalid. Error: {error_message} ---")


    print("--- [app.py] index() - Rendering template. ---")
    return render_template("index.html", features_info=FEATURES_INFO, ordered_features=ORDERED_FEATURES,
                           prediction_text=prediction_text, error_message=error_message, form_values=form_values)


print("--- [app.py] Checking if __name__ == '__main__' ... ---")
if __name__ == "__main__":
    print("--- [app.py] __name__ is '__main__'. Starting Flask development server... ---")
    try:
        app.run(debug=True)
    except Exception as e_app_run:
        print(f"--- [app.py] CRITICAL ERROR starting Flask app.run: {e_app_run} ---")
        sys.exit(f"Error starting Flask app.run: {e_app_run}")
else:
    print(f"--- [app.py] __name__ is '{__name__}'. Flask server will not start automatically (e.g. when imported). ---")

print("--- [app.py] Script execution reached end (should not happen if app.run starts). ---")