import os
import pickle
import numpy as np
from flask import Flask, render_template, request

# ============================
# Load ML Models
# ============================

# Models are in the SAME folder as app.py
heart_model = pickle.load(open("heart_best_model.pkl", "rb"))

# Liver model (XGBoost-based) – may show warnings but should load
liver_model = pickle.load(open("liver_best_model.pkl", "rb"))

# Kidney model – may fail due to sklearn version incompatibility
try:
    kidney_model = pickle.load(open("kidney_best_model.pkl", "rb"))
    print("Kidney model loaded successfully.")
except Exception as e:
    kidney_model = None
    print("WARNING: Kidney model could not be loaded:", e)

# Cancer model – may also fail due to sklearn version incompatibility
try:
    cancer_model = pickle.load(open("cancer_best_model.pkl", "rb"))
    print("Cancer model loaded successfully.")
except Exception as e:
    cancer_model = None
    print("WARNING: Cancer model could not be loaded:", e)

# ============================
# Flask App
# ============================

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# ============================
# Routes
# ============================


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result-history", methods=["GET", "POST"])
def result_history():
    """
    Database is disabled for now.
    We return empty lists so the template still renders clean tables.
    """
    return render_template(
        "result_history.html",
        heart=[],
        liver=[],
        cancer=[],
        kidney=[]
    )


@app.route("/cancer", methods=["GET", "POST"])
def cancer():
    return render_template("cancer.html")


@app.route("/heart", methods=["GET", "POST"])
def heart():
    return render_template("heart.html")


@app.route("/kidney", methods=["GET", "POST"])
def kidney():
    return render_template("kidney.html")


@app.route("/liver", methods=["GET", "POST"])
def liver():
    return render_template("liver.html")
# ============================
# Prediction Endpoint
# ============================

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method != "POST":
        return render_template("predict.html", prediction=None)

    # Use number of features to detect which form was submitted
    values = [float(x) for x in request.form.values()]
    n_features = len(values)

    # --------------------------------
    # LIVER (10 Features)
    # --------------------------------
    if n_features == 10:
        Age = int(request.form["Age"])
        Total_Bilirubin = float(request.form["Total_Bilirubin"])
        Direct_Bilirubin = float(request.form["Direct_Bilirubin"])
        Alkaline_Phosphotase = int(request.form["Alkaline_Phosphotase"])
        Alamine_Aminotransferase = int(request.form["Alamine_Aminotransferase"])
        Aspartate_Aminotransferase = int(request.form["Aspartate_Aminotransferase"])
        Total_Protiens = float(request.form["Total_Protiens"])
        Albumin = float(request.form["Albumin"])
        Albumin_and_Globulin_Ratio = float(request.form["Albumin_and_Globulin_Ratio"])
        Gender_Male = int(request.form["Gender_Male"])

        data = np.array([[
            Age,
            Total_Bilirubin,
            Direct_Bilirubin,
            Alkaline_Phosphotase,
            Alamine_Aminotransferase,
            Aspartate_Aminotransferase,
            Total_Protiens,
            Albumin,
            Albumin_and_Globulin_Ratio,
            Gender_Male
        ]])

        my_prediction = liver_model.predict(data)  # array([0]) or array([1])
        return render_template("predict.html", prediction=int(my_prediction[0]))

    # --------------------------------
    # HEART (13 Features)
    # --------------------------------
    if n_features == 13:
        age = int(request.form["age"])
        sex = int(request.form["sex"])
        cp = int(request.form["cp"])
        trestbps = int(request.form["trestbps"])
        chol = int(request.form["chol"])
        fbs = int(request.form["fbs"])
        restecg = int(request.form["restecg"])
        thalach = int(request.form["thalach"])
        exang = int(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = int(request.form["slope"])
        ca = int(request.form["ca"])
        thal = int(request.form["thal"])

        data = np.array([[
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]])

        my_prediction = heart_model.predict(data)
        return render_template("predict.html", prediction=int(my_prediction[0]))

    # --------------------------------
    # KIDNEY (18 Features)
    # --------------------------------
    if n_features == 18:
        # If kidney model failed to load, don't crash – just show empty result
        if kidney_model is None:
            print("Kidney model is not available in this environment.")
            return render_template("predict.html", prediction=None)

        age = float(request.form["age"])
        bp = float(request.form["bp"])
        al = float(request.form["al"])
        su = float(request.form["su"])
        rbc = int(request.form["rbc"])
        pc = int(request.form["pc"])
        pcc = int(request.form["pcc"])
        ba = int(request.form["ba"])
        bgr = float(request.form["bgr"])
        bu = float(request.form["bu"])
        sc = float(request.form["sc"])
        pot = float(request.form["pot"])
        wc = int(request.form["wc"])
        htn = int(request.form["htn"])
        dm = int(request.form["dm"])
        cad = int(request.form["cad"])
        pe = int(request.form["pe"])
        ane = int(request.form["ane"])

        data = np.array([[
            age, bp, al, su, rbc, pc, pcc, ba,
            bgr, bu, sc, pot, wc, htn, dm, cad, pe, ane
        ]])

        my_prediction = kidney_model.predict(data)
        return render_template("predict.html", prediction=int(my_prediction[0]))

    # --------------------------------
    # CANCER (30 expected features – we currently collect 26)
    # --------------------------------
    if n_features == 26:
        # If cancer model failed to load, don't crash – just show empty result
        if cancer_model is None:
            print("Cancer model is not available in this environment.")
            return render_template("predict.html", prediction=None)

        # Helper that handles both underscore and space versions
        def get_float(*keys):
            for k in keys:
                val = request.form.get(k)
                if val is not None and val != "":
                    return float(val)
            raise KeyError(f"None of these keys found in form: {keys}")

        # Features that we DO collect from the form
        radius_mean             = get_float("radius_mean")
        texture_mean            = get_float("texture_mean")
        perimeter_mean          = get_float("perimeter_mean")
        area_mean               = get_float("area_mean")
        smoothness_mean         = get_float("smoothness_mean")
        compactness_mean        = get_float("compactness_mean")
        concavity_mean          = get_float("concavity_mean")
        concave_points_mean     = get_float("concave_points_mean", "concave points_mean")
        symmetry_mean           = get_float("symmetry_mean")

        radius_se               = get_float("radius_se")
        perimeter_se            = get_float("perimeter_se")
        area_se                 = get_float("area_se")
        compactness_se          = get_float("compactness_se")
        concavity_se            = get_float("concavity_se")
        concave_points_se       = get_float("concave_points_se", "concave points_se")
        fractal_dimension_se    = get_float("fractal_dimension_se")

        radius_worst            = get_float("radius_worst")
        texture_worst           = get_float("texture_worst")
        perimeter_worst         = get_float("perimeter_worst")
        area_worst              = get_float("area_worst")
        smoothness_worst        = get_float("smoothness_worst")
        compactness_worst       = get_float("compactness_worst")
        concavity_worst         = get_float("concavity_worst")
        concave_points_worst    = get_float("concave_points_worst", "concave points_worst")
        symmetry_worst          = get_float("symmetry_worst")
        fractal_dimension_worst = get_float("fractal_dimension_worst")

        # ---- Missing 4 features in the HTML form ----
        # texture_se, smoothness_se, symmetry_se, fractal_dimension_mean
        # We fill them with 0.0 so the model always gets 30 features.
        raw_texture_se = request.form.get("texture_se")
        texture_se = float(raw_texture_se) if raw_texture_se not in (None, "") else 0.0

        raw_smoothness_se = request.form.get("smoothness_se")
        smoothness_se = float(raw_smoothness_se) if raw_smoothness_se not in (None, "") else 0.0

        raw_symmetry_se = request.form.get("symmetry_se")
        symmetry_se = float(raw_symmetry_se) if raw_symmetry_se not in (None, "") else 0.0

        raw_fd_mean = request.form.get("fractal_dimension_mean")
        fractal_dimension_mean = float(raw_fd_mean) if raw_fd_mean not in (None, "") else 0.0

        # Build the vector in the standard 30-feature order:
        # All *_mean, then *_se, then *_worst
        data = np.array([[
            # means (10)
            radius_mean,
            texture_mean,
            perimeter_mean,
            area_mean,
            smoothness_mean,
            compactness_mean,
            concavity_mean,
            concave_points_mean,
            symmetry_mean,
            fractal_dimension_mean,

            # standard errors (10)
            radius_se,
            texture_se,
            perimeter_se,
            area_se,
            smoothness_se,
            compactness_se,
            concavity_se,
            concave_points_se,
            symmetry_se,
            fractal_dimension_se,

            # worst (10)
            radius_worst,
            texture_worst,
            perimeter_worst,
            area_worst,
            smoothness_worst,
            compactness_worst,
            concavity_worst,
            concave_points_worst,
            symmetry_worst,
            fractal_dimension_worst
        ]])

        # Predict
        my_prediction = cancer_model.predict(data)
        pred_int = int(my_prediction[0]) if hasattr(my_prediction, "__len__") else int(my_prediction)

        return render_template("predict.html", prediction=pred_int)

    # Fallback: unknown form shape
    return render_template("predict.html", prediction=None)


# ============================
# Main
# ============================

if __name__ == "__main__":
    app.run(debug=True)
