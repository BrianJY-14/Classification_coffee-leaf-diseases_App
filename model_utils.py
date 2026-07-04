from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "VGG16_transfer_best.weights.h5"
IMG_SIZE = (100, 100)
NUM_CLASSES = 5

CLASS_NAMES = [
    "Cercospora",
    "Healthy",
    "Leaf rust",
    "Miner",
    "Phoma",
]

CLASS_DETAILS = {
    "Cercospora": {
        "name_es": "Cercospora",
        "tone": "Riesgo de mancha foliar por Cercospora.",
        "hint": "Revisa manchas circulares, bordes definidos y amarillamiento alrededor de la lesion.",
    },
    "Healthy": {
        "name_es": "Hoja sana",
        "tone": "No se detecta una enfermedad entre las clases entrenadas.",
        "hint": "Mantén monitoreo preventivo, buena ventilacion y nutricion balanceada.",
    },
    "Leaf rust": {
        "name_es": "Roya del cafeto",
        "tone": "Patron compatible con roya.",
        "hint": "Busca pustulas anaranjadas en el enves y evita humedad persistente en el follaje.",
    },
    "Miner": {
        "name_es": "Minador de hoja",
        "tone": "Patron compatible con dano por minador.",
        "hint": "Observa galerias o tuneles claros dentro del tejido de la hoja.",
    },
    "Phoma": {
        "name_es": "Phoma",
        "tone": "Patron compatible con Phoma.",
        "hint": "Contrasta con necrosis, manchas oscuras y avance en bordes o lesiones.",
    },
}


def _import_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers
        from tensorflow.keras.applications import VGG16
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow no esta instalado. Ejecuta: pip install -r requirements.txt"
        ) from exc

    return tf, layers, VGG16


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo de pesos en: {MODEL_PATH}"
        )

    tf, layers, VGG16 = _import_tensorflow()
    tf.keras.backend.clear_session()

    inputs = layers.Input(shape=(100, 100, 3), name="input_rgb_100")
    x = layers.Lambda(
        lambda z: tf.keras.applications.vgg16.preprocess_input(z * 255.0),
        output_shape=(100, 100, 3),
        name="vgg16_preprocess",
    )(inputs)

    vgg_base = VGG16(
        include_top=False,
        weights=None,
        input_tensor=x,
    )
    vgg_base.trainable = False

    x = layers.Flatten(name="flatten_vgg16")(vgg_base.output)
    x = layers.Dense(46, activation="relu", name="dense_46")(x)
    x = layers.Dense(40, activation="relu", name="dense_40")(x)
    x = layers.Dense(10, activation="relu", name="dense_10")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="softmax_5")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="VGG16_Baht")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )
    model.load_weights(MODEL_PATH)
    return model


def preprocess_image(image_stream):
    try:
        image = Image.open(image_stream)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("El archivo no parece ser una imagen valida.") from exc

    image = image.resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict_leaf_image(image_stream):
    model = load_model()
    x = preprocess_image(image_stream)
    probs = model.predict(x, verbose=0)[0].astype(float)
    pred_id = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_id]
    confidence = float(probs[pred_id])

    probabilities = [
        {
            "class_name": class_name,
            "display_name": CLASS_DETAILS[class_name]["name_es"],
            "probability": float(prob),
            "percent": round(float(prob) * 100, 2),
        }
        for class_name, prob in zip(CLASS_NAMES, probs)
    ]

    return {
        "model": "VGG16_transfer",
        "pred_id": pred_id,
        "pred_label": pred_label,
        "display_name": CLASS_DETAILS[pred_label]["name_es"],
        "confidence": confidence,
        "confidence_percent": round(confidence * 100, 2),
        "message": CLASS_DETAILS[pred_label]["tone"],
        "hint": CLASS_DETAILS[pred_label]["hint"],
        "probabilities": probabilities,
    }
