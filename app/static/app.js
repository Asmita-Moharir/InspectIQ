
const input = document.getElementById("image-input");
const dropZone = document.getElementById("drop-zone");
const preview = document.getElementById("image-preview");
const placeholder = document.getElementById("upload-placeholder");
const analyzeBtn = document.getElementById("analyze-btn");
const message = document.getElementById("message");

const emptyState = document.getElementById("empty-state");
const results = document.getElementById("results");
const predictedClass = document.getElementById("predicted-class");
const confidence = document.getElementById("confidence");
const confidenceBar = document.getElementById("confidence-bar");
const predictionsList = document.getElementById("predictions-list");
const downloadBtn = document.getElementById("download-btn");

const systemStatus = document.getElementById("system-status");
const statusDot = document.querySelector(".status-dot");

const MAX_SIZE = 10 * 1024 * 1024;
let selectedFile = null;
let lastResult = null;
let previewUrl = null;


async function checkHealth() {
    try {
        const response = await fetch("/api/health");
        const body = await response.text();

        if (!response.ok) {
            throw new Error(
                `Health check failed: HTTP ${response.status} — ${body}`
            );
        }

        const data = JSON.parse(body);

        systemStatus.textContent = `Model online · ${data.device}`;
        statusDot.classList.add("online");
        statusDot.classList.remove("offline");
    } catch (error) {
        systemStatus.textContent = "Health check failed";
        statusDot.classList.add("offline");
        statusDot.classList.remove("online");
        console.error("InspectIQ health check:", error);
    }
}


function setMessage(text, isError = false) {
    message.textContent = text;
    message.classList.toggle("error", isError);
}

function clearResults() {
    lastResult = null;
    results.classList.add("hidden");
    emptyState.classList.remove("hidden");
}

function chooseFile(file) {
    if (!file) return;

    const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/webp",
    ];

    if (!allowedTypes.includes(file.type)) {
        setMessage("Choose a JPG, PNG, or WebP image.", true);
        return;
    }

    if (file.size > MAX_SIZE) {
        setMessage("The image must be smaller than 10 MB.", true);
        return;
    }

    selectedFile = file;
    clearResults();

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);

    preview.src = previewUrl;
    preview.classList.remove("hidden");
    placeholder.classList.add("hidden");

    analyzeBtn.disabled = false;
    setMessage(`Selected: ${file.name}`);
}

dropZone.addEventListener("click", () => input.click());

dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        input.click();
    }
});

input.addEventListener("change", () => {
    chooseFile(input.files[0]);
});

dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
    chooseFile(event.dataTransfer.files[0]);
});

function renderPredictions(data) {
    predictedClass.textContent =
        data.predicted_class.replaceAll("_", " ").replace(
            /\b\w/g,
            (char) => char.toUpperCase()
        );

    confidence.textContent = `${data.confidence.toFixed(2)}%`;
    confidenceBar.style.width = `${data.confidence}%`;

    predictionsList.replaceChildren();

    for (const item of data.predictions) {
        const row = document.createElement("div");
        row.className = "prediction";

        const top = document.createElement("div");
        top.className = "prediction-top";

        const name = document.createElement("span");
        name.textContent = item.class_name.replaceAll("_", " ");

        const value = document.createElement("span");
        value.textContent = `${item.confidence.toFixed(2)}%`;

        const track = document.createElement("div");
        track.className = "prediction-track";

        const fill = document.createElement("div");
        fill.className = "prediction-fill";
        fill.style.width = `${item.confidence}%`;

        top.append(name, value);
        track.append(fill);
        row.append(top, track);
        predictionsList.append(row);
    }

    emptyState.classList.add("hidden");
    results.classList.remove("hidden");
}

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing...";
    setMessage("Running inference on the selected image.");

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed.");
        }

        lastResult = data;
        renderPredictions(data);
        setMessage("Analysis completed successfully.");
    } catch (error) {
        setMessage(error.message, true);
    } finally {
        analyzeBtn.disabled = !selectedFile;
        analyzeBtn.innerHTML = 'Analyze surface <span>→</span>';
    }
});

downloadBtn.addEventListener("click", () => {
    if (!lastResult) return;

    const blob = new Blob(
        [JSON.stringify(lastResult, null, 2)],
        { type: "application/json" }
    );

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "inspectiq-result.json";
    link.click();

    URL.revokeObjectURL(url);
});

checkHealth();
