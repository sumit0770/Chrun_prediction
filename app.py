from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Load model and pipeline
try:
    model = joblib.load("model/model.pkl")
    pipeline = joblib.load("model/pipeline.pkl")
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

@app.route("/", methods=["GET", "POST"])
def home():
    probability = None
    risk = None
    risk_color = None
    error = None
    current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    if request.method == "POST":
        if not model_loaded:
            error = "Model not loaded. Please check server configuration."
        else:
            try:
                # Get form data
                data = {
                    "gender": request.form["gender"],
                    "SeniorCitizen": int(request.form["SeniorCitizen"]),
                    "Partner": request.form["Partner"],
                    "Dependents": request.form["Dependents"],
                    "tenure": int(request.form["tenure"]),
                    "PhoneService": request.form["PhoneService"],
                    "MultipleLines": request.form["MultipleLines"],
                    "InternetService": request.form["InternetService"],
                    "OnlineSecurity": request.form["OnlineSecurity"],
                    "OnlineBackup": request.form["OnlineBackup"],
                    "DeviceProtection": request.form["DeviceProtection"],
                    "TechSupport": request.form["TechSupport"],
                    "StreamingTV": request.form["StreamingTV"],
                    "StreamingMovies": request.form["StreamingMovies"],
                    "Contract": request.form["Contract"],
                    "PaperlessBilling": request.form["PaperlessBilling"],
                    "PaymentMethod": request.form["PaymentMethod"],
                    "MonthlyCharges": float(request.form["MonthlyCharges"]),
                    "TotalCharges": float(request.form["TotalCharges"])
                }

                # Create DataFrame and make prediction
                df = pd.DataFrame([data])
                X = pipeline.transform(df)
                prob = model.predict_proba(X)[0][1]
                
                # Format probability as percentage
                probability = f"{prob:.1%}"
                
                # Determine risk level and color
                if prob > 0.6:
                    risk = "High"
                    risk_color = "#e53e3e"
                elif prob > 0.3:
                    risk = "Medium"
                    risk_color = "#dd6b20"
                else:
                    risk = "Low"
                    risk_color = "#38a169"

                return render_template(
                    "index.html", 
                    probability=probability,
                    risk=risk,
                    risk_color=risk_color,
                    current_time=current_time,
                    error=error
                )

            except KeyError as e:
                error = f"Missing required field: {str(e)}"
            except ValueError as e:
                error = f"Invalid input value: {str(e)}"
            except Exception as e:
                error = f"Prediction error: {str(e)}"
                print(f"Error: {e}")

    return render_template(
        "index.html", 
        probability=probability,
        risk=risk,
        risk_color=risk_color,
        current_time=current_time,
        error=error
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)