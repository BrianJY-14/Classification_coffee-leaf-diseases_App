const dropzone = document.getElementById("dropzone");
const imageInput = document.getElementById("imageInput");
const previewStage = document.getElementById("previewStage");
const previewImage = document.getElementById("previewImage");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const modelStatus = document.getElementById("modelStatus");
const resultPanel = document.querySelector(".result-panel");
const resultLabel = document.getElementById("resultLabel");
const confidenceValue = document.getElementById("confidenceValue");
const resultMessage = document.getElementById("resultMessage");
const probabilityList = document.getElementById("probabilityList");
const hintBox = document.getElementById("hintBox");

let selectedFile = null;
let previewUrl = null;

function setStatus(text) {
  modelStatus.textContent = text;
}

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("Imagen no valida");
    return;
  }

  selectedFile = file;

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewStage.classList.add("has-image");
  resultPanel.classList.remove("result-ready");
  predictBtn.disabled = false;
  setStatus("Imagen cargada");
}

function resetResult() {
  resultLabel.textContent = "Esperando imagen";
  confidenceValue.textContent = "0%";
  resultMessage.textContent = "El panel mostrara la clase estimada y la confianza del modelo.";
  hintBox.textContent = "Diagnostico automatico de apoyo. Verifica el resultado con inspeccion agronomica.";

  probabilityList.querySelectorAll(".probability-row").forEach((row) => {
    row.querySelector(".probability-meta strong").textContent = "0%";
    row.querySelector(".bar-track span").style.width = "0%";
  });
}

function clearFile() {
  selectedFile = null;
  imageInput.value = "";

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  previewImage.removeAttribute("src");
  previewStage.classList.remove("has-image", "scanning");
  resultPanel.classList.remove("result-ready");
  predictBtn.disabled = true;
  setStatus("Modelo listo");
  resetResult();
}

function setLoading(isLoading) {
  previewStage.classList.toggle("scanning", isLoading);
  predictBtn.disabled = isLoading || !selectedFile;
  predictBtn.querySelector("span").textContent = isLoading ? "Analizando" : "Predecir";
  setStatus(isLoading ? "Escaneando" : "Modelo listo");
}

function renderProbabilities(probabilities) {
  probabilityList.innerHTML = "";

  probabilities.forEach((item) => {
    const row = document.createElement("div");
    row.className = "probability-row";
    row.innerHTML = `
      <div class="probability-meta">
        <span>${item.display_name}</span>
        <strong>${item.percent.toFixed(2)}%</strong>
      </div>
      <div class="bar-track"><span style="width: 0%"></span></div>
    `;
    probabilityList.appendChild(row);

    requestAnimationFrame(() => {
      row.querySelector(".bar-track span").style.width = `${Math.max(0, Math.min(100, item.percent))}%`;
    });
  });
}

function renderResult(data) {
  resultPanel.classList.add("result-ready");
  resultLabel.textContent = data.display_name;
  confidenceValue.textContent = `${data.confidence_percent.toFixed(2)}%`;
  resultMessage.textContent = data.message;
  hintBox.textContent = data.hint;
  renderProbabilities(data.probabilities);
}

async function predict() {
  if (!selectedFile) {
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedFile);
  setLoading(true);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "No se pudo completar la prediccion.");
    }

    window.setTimeout(() => {
      renderResult(data);
      setLoading(false);
      setStatus("Prediccion lista");
    }, 720);
  } catch (error) {
    setLoading(false);
    setStatus("Error");
    resultLabel.textContent = "Sin resultado";
    confidenceValue.textContent = "--";
    resultMessage.textContent = error.message;
    hintBox.textContent = "Revisa que el archivo sea una imagen y que TensorFlow este instalado.";
  }
}

imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  setFile(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("drag-over");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag-over");
  });
});

dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  setFile(file);
});

predictBtn.addEventListener("click", predict);
clearBtn.addEventListener("click", clearFile);
