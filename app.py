from pathlib import Path

from flask import Flask, jsonify, render_template, request

from model_utils import CLASS_NAMES, MODEL_PATH, predict_leaf_image


app = Flask(__name__, static_folder="public/static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html", classes=CLASS_NAMES)


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_file": MODEL_PATH.name,
            "model_exists": Path(MODEL_PATH).exists(),
            "classes": CLASS_NAMES,
        }
    )


@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No se recibio ninguna imagen."}), 400

    image_file = request.files["image"]

    if not image_file.filename:
        return jsonify({"error": "Selecciona una imagen valida."}), 400

    try:
        result = predict_leaf_image(image_file.stream)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "No se pudo analizar la imagen."}), 500

    return jsonify(result)


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "La imagen supera el limite de 4 MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
