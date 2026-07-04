# Classification Coffee Leaf Diseases App

Aplicacion web para clasificar imagenes de hojas de cafeto con el modelo `VGG16_transfer`.

## Clases

- Cercospora
- Healthy
- Leaf rust
- Miner
- Phoma

## Ejecutar en local

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Luego abre:

```text
http://127.0.0.1:5000
```

> En Windows, si TensorFlow falla por rutas largas al instalar dentro de una
> carpeta profunda, clona la repo en una ruta corta como `C:\coffee-app` o
> habilita Windows Long Paths.

## Modelo

La app reconstruye la arquitectura VGG16 usada en el notebook:

- Entrada `100x100x3`
- Preprocesamiento ImageNet para VGG16
- Base `VGG16(include_top=False)`
- `Flatten`
- `Dense(46, relu)`
- `Dense(40, relu)`
- `Dense(10, relu)`
- `Dense(5, softmax)`

El archivo de pesos esperado es:

```text
models/VGG16_transfer_best.weights.h5
```

## Aviso

El resultado es una prediccion automatica de apoyo. Para decisiones de manejo agricola, confirma el diagnostico con inspeccion de campo o asesoria agronomica.
