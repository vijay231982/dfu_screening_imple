import random
import time
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.applications import densenet, efficientnet, mobilenet_v2
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploaded_images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

MODEL_SPECS = [
    {
        "name": "MobileNetV2",
        "optimizer_name": "Adam",
        "optimizer": lambda: Adam(learning_rate=1e-4),
        "builder": lambda: mobilenet_v2.MobileNetV2(weights="imagenet"),
        "preprocess": mobilenet_v2.preprocess_input,
    },
    {
        "name": "EfficientNetB0",
        "optimizer_name": "SGD",
        "optimizer": lambda: SGD(learning_rate=1e-4, momentum=0.9),
        "builder": lambda: efficientnet.EfficientNetB0(weights="imagenet"),
        "preprocess": efficientnet.preprocess_input,
    },
    {
        "name": "DenseNet121",
        "optimizer_name": "RMSprop",
        "optimizer": lambda: RMSprop(learning_rate=1e-4),
        "builder": lambda: densenet.DenseNet121(weights="imagenet"),
        "preprocess": densenet.preprocess_input,
    },
]


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
MODEL_CACHE = {}
ABNORMAL_KEYWORDS = {
    "mask",
    "spotlight",
    "rule",
    "bandage",
    "mosquito",
    "coral",
    "fungus",
    "burrito",
    "scab",
    "stain",
    "sunscreen",
    "face powder",
}
NORMAL_KEYWORDS = {
    "lotion",
    "soap",
    "face powder",
    "shirt",
    "jersey",
    "apron",
    "swimming trunks",
    "bath towel",
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_unique_path(filename: str) -> Path:
    candidate = UPLOAD_DIR / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        candidate = UPLOAD_DIR / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def get_model(spec: dict):
    model_name = spec["name"]
    if model_name not in MODEL_CACHE:
        model = spec["builder"]()
        model.compile(optimizer=spec["optimizer"](), loss="categorical_crossentropy")
        MODEL_CACHE[model_name] = model
    return MODEL_CACHE[model_name]


def prepare_image(image_path: Path) -> np.ndarray:
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    return np.expand_dims(img_array, axis=0)


def decode_top_predictions(predictions: np.ndarray, top: int = 3) -> list[dict[str, float | str]]:
    decoded = mobilenet_v2.decode_predictions(predictions, top=top)[0]
    return [
        {
            "label": label.replace("_", " ").title(),
            "confidence": float(score) * 100,
            "class_id": class_id,
        }
        for class_id, label, score in decoded
    ]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


def compute_visual_features(image_path: Path) -> dict[str, float]:
    img = keras_image.load_img(image_path, target_size=(224, 224))
    pixels = keras_image.img_to_array(img).astype("float32") / 255.0

    red_channel = pixels[:, :, 0]
    green_channel = pixels[:, :, 1]
    blue_channel = pixels[:, :, 2]
    grayscale = np.mean(pixels, axis=2)

    redness = float(np.mean(np.clip(red_channel - ((green_channel + blue_channel) / 2), 0.0, 1.0)))
    contrast = float(np.std(grayscale))
    dark_ratio = float(np.mean(grayscale < 0.28))
    bright_ratio = float(np.mean(grayscale > 0.78))

    grad_y, grad_x = np.gradient(grayscale)
    edge_density = float(np.mean(np.sqrt((grad_x ** 2) + (grad_y ** 2))))

    return {
        "redness": redness,
        "contrast": contrast,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "edge_density": edge_density,
    }


def keyword_signal(top_predictions: list[dict[str, float | str]]) -> float:
    score = 0.0
    for prediction in top_predictions:
        label = str(prediction["label"]).lower()
        confidence = float(prediction["confidence"]) / 100.0
        if any(keyword in label for keyword in ABNORMAL_KEYWORDS):
            score += 0.35 * confidence
        if any(keyword in label for keyword in NORMAL_KEYWORDS):
            score -= 0.2 * confidence
    return score


def candidate_skin_probability(candidate: dict, visual_features: dict[str, float]) -> float:
    top_confidence = float(candidate["top_confidence"]) / 100.0
    uncertainty = 1.0 - top_confidence
    feature_score = (
        (visual_features["redness"] * 3.0)
        + (visual_features["contrast"] * 1.8)
        + (visual_features["dark_ratio"] * 1.4)
        + (visual_features["edge_density"] * 2.2)
        - (visual_features["bright_ratio"] * 0.8)
    )
    raw_score = (
        -1.15
        + feature_score
        + (uncertainty * 0.9)
        + keyword_signal(candidate["top_predictions"])
    )
    return float(sigmoid(raw_score))


def run_candidate_models(image_path: Path) -> tuple[list[dict], float]:
    raw_image = prepare_image(image_path)
    candidates = []
    total_started_at = time.perf_counter()
    visual_features = compute_visual_features(image_path)

    for spec in MODEL_SPECS:
        model = get_model(spec)
        processed_image = spec["preprocess"](raw_image.copy())

        started_at = time.perf_counter()
        predictions = model.predict(processed_image, verbose=0)
        elapsed_ms = (time.perf_counter() - started_at) * 1000

        top_predictions = decode_top_predictions(predictions, top=3)
        candidates.append(
            {
                "model_name": spec["name"],
                "optimizer_name": spec["optimizer_name"],
                "predictions": predictions[0],
                "top_predictions": top_predictions,
                "top_label": top_predictions[0]["label"],
                "top_confidence": top_predictions[0]["confidence"],
                "inference_ms": elapsed_ms,
            }
        )

    for candidate in candidates:
        abnormal_probability = candidate_skin_probability(candidate, visual_features)
        candidate["abnormal_probability"] = abnormal_probability * 100
        candidate["normal_probability"] = (1.0 - abnormal_probability) * 100

    total_elapsed_ms = (time.perf_counter() - total_started_at) * 1000
    return candidates, total_elapsed_ms, visual_features


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    positive_weights = np.clip(weights.astype("float32"), 0.0, None)
    weight_sum = float(np.sum(positive_weights))
    if weight_sum <= 0:
        return np.full_like(positive_weights, 1.0 / len(positive_weights))
    return positive_weights / weight_sum


def ensemble_fitness(weights: np.ndarray, candidates: list[dict]) -> float:
    normalized = normalize_weights(weights)
    stacked = np.stack([candidate["predictions"] for candidate in candidates], axis=0)
    ensemble_probs = np.sum(stacked * normalized[:, None], axis=0)

    top_indices = np.argsort(ensemble_probs)[-3:][::-1]
    top_score = float(ensemble_probs[top_indices[0]])
    second_score = float(ensemble_probs[top_indices[1]])
    margin = top_score - second_score

    selected = [candidates[i] for i, value in enumerate(normalized) if value > 0.1]
    label_agreement = sum(
        1 for candidate in selected if candidate["top_label"] == candidates[np.argmax(normalized)]["top_label"]
    )
    agreement_score = label_agreement / max(len(selected), 1)

    active_penalty = np.count_nonzero(normalized > 0.1) * 0.02
    return top_score + (margin * 1.5) + (agreement_score * 0.1) - active_penalty


def genetic_ensemble_selection(candidates: list[dict]) -> dict:
    rng = random.Random(42)
    candidate_count = len(candidates)
    population_size = 12
    generations = 8
    mutation_rate = 0.2

    population = [
        normalize_weights(np.array([rng.random() for _ in range(candidate_count)], dtype="float32"))
        for _ in range(population_size)
    ]

    best_weights = population[0]
    best_fitness = float("-inf")

    for _ in range(generations):
        scored = sorted(
            ((ensemble_fitness(weights, candidates), weights) for weights in population),
            key=lambda item: item[0],
            reverse=True,
        )

        if scored[0][0] > best_fitness:
            best_fitness = scored[0][0]
            best_weights = scored[0][1]

        survivors = [weights for _, weights in scored[:4]]
        next_population = survivors.copy()

        while len(next_population) < population_size:
            parent_a = survivors[rng.randrange(len(survivors))]
            parent_b = survivors[rng.randrange(len(survivors))]
            blend = rng.random()
            child = normalize_weights((blend * parent_a) + ((1 - blend) * parent_b))

            if rng.random() < mutation_rate:
                mutation = np.array(
                    [rng.uniform(-0.15, 0.15) for _ in range(candidate_count)],
                    dtype="float32",
                )
                child = normalize_weights(child + mutation)

            next_population.append(child)

        population = next_population

    stacked = np.stack([candidate["predictions"] for candidate in candidates], axis=0)
    normalized = normalize_weights(best_weights)
    ensemble_probs = np.sum(stacked * normalized[:, None], axis=0, keepdims=True)
    top_predictions = decode_top_predictions(ensemble_probs, top=3)

    members = []
    for index, candidate in enumerate(candidates):
        if normalized[index] > 0.1:
            members.append(
                {
                    "model_name": candidate["model_name"],
                    "optimizer_name": candidate["optimizer_name"],
                    "weight": float(normalized[index]) * 100,
                    "top_label": candidate["top_label"],
                }
            )

    return {
        "top_predictions": top_predictions,
        "members": members,
        "fitness": best_fitness,
    }


def build_skin_assessment(
    candidates: list[dict],
    ensemble: dict,
    visual_features: dict[str, float],
) -> dict:
    abnormal_probability = 0.0
    for index, candidate in enumerate(candidates):
        candidate_weight = 0.0
        for member in ensemble["members"]:
            if member["model_name"] == candidate["model_name"]:
                candidate_weight = float(member["weight"]) / 100.0
                break
        abnormal_probability += candidate_weight * (float(candidate["abnormal_probability"]) / 100.0)

    if abnormal_probability == 0.0:
        abnormal_probability = float(np.mean([candidate["abnormal_probability"] for candidate in candidates])) / 100.0

    visual_adjustment = (
        (visual_features["redness"] * 0.22)
        + (visual_features["dark_ratio"] * 0.18)
        + (visual_features["edge_density"] * 0.24)
        - (visual_features["bright_ratio"] * 0.08)
    )
    abnormal_probability = float(np.clip((abnormal_probability * 0.75) + visual_adjustment, 0.01, 0.99))
    normal_probability = 1.0 - abnormal_probability
    label = "Abnormal Skin" if abnormal_probability >= 0.5 else "Normal Skin"

    return {
        "label": label,
        "abnormal_probability": abnormal_probability * 100,
        "normal_probability": normal_probability * 100,
        "visual_features": {
            "redness": visual_features["redness"] * 100,
            "contrast": visual_features["contrast"] * 100,
            "dark_ratio": visual_features["dark_ratio"] * 100,
            "edge_density": visual_features["edge_density"] * 100,
        },
    }


def analyze_uploaded_image(image_path: Path) -> tuple[list[dict], dict, dict, float]:
    candidates, total_elapsed_ms, visual_features = run_candidate_models(image_path)
    ensemble = genetic_ensemble_selection(candidates)
    skin_assessment = build_skin_assessment(candidates, ensemble, visual_features)
    return candidates, ensemble, skin_assessment, total_elapsed_ms


@app.route("/uploaded_images/<path:filename>")
def uploaded_image(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/", methods=["GET", "POST"])
def index():
    message = None
    error = None
    saved_file = None
    image_url = None
    candidate_results = []
    ensemble_result = None
    skin_assessment = None
    inference_ms = None

    if request.method == "POST":
        image = request.files.get("image")

        if image is None or image.filename == "":
            error = "Select an image to upload."
        elif not allowed_file(image.filename):
            error = "Only image files are allowed."
        else:
            UPLOAD_DIR.mkdir(exist_ok=True)
            filename = secure_filename(image.filename)
            save_path = make_unique_path(filename)
            image.save(save_path)
            saved_file = save_path.name
            image_url = f"/uploaded_images/{saved_file}"

            try:
                candidate_results, ensemble_result, skin_assessment, inference_ms = analyze_uploaded_image(save_path)
                message = "Image uploaded and processed with optimizer-aware ensemble skin classification."
            except Exception:
                error = "Image was uploaded, but ensemble processing failed."

    return render_template(
        "index.html",
        message=message,
        error=error,
        saved_file=saved_file,
        upload_dir=UPLOAD_DIR.name,
        image_url=image_url,
        candidate_results=candidate_results,
        ensemble_result=ensemble_result,
        skin_assessment=skin_assessment,
        inference_ms=inference_ms,
    )


if __name__ == "__main__":
    app.run(debug=True)
