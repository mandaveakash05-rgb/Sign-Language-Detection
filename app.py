"""Train and run an ASL alphabet classifier and real-time hand sign detection."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
import time
from pathlib import Path

# Suppress TensorFlow informational logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False


ASL_LABELS = tuple(letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if letter not in "JZ")
DATASET_LABEL_CODES = tuple(range(9)) + tuple(range(10, 25))
LABEL_CODE_TO_INDEX = {
    label_code: index for index, label_code in enumerate(DATASET_LABEL_CODES)
}
IMAGE_SIDE = 28
PIXEL_COUNT = IMAGE_SIDE * IMAGE_SIDE
DEFAULT_DATA_DIR = Path("datasets")
DEFAULT_MODEL_PATH = Path("models") / "sign_language_model.joblib"
HAND_TASK_PATH = Path("models") / "hand_landmarker.task"
COMMON_WORDS = ("HI", "BYE", "LIKE", "HELLO", "HELP", "THANKS", "YES", "NO")

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                               # Palm base
)

GESTURE_DICTIONARY = [
    {
        "code": "HELLO",
        "letter": "5",
        "sign": "Open Palm (5)",
        "message": "Hello / Welcome / Stop",
        "action": "All 5 fingers spread open wide",
    },
    {
        "code": "PEACE",
        "letter": "V",
        "sign": "Peace / V Sign",
        "message": "Peace & Good Health",
        "action": "Index & Middle up in V shape",
    },
    {
        "code": "YES",
        "letter": "A",
        "sign": "Thumbs Up",
        "message": "Yes / Approved / Good",
        "action": "Thumb pointing straight up",
    },
    {
        "code": "NO",
        "letter": "A",
        "sign": "Thumbs Down",
        "message": "No / Disagree / Rejected",
        "action": "Thumb pointing down",
    },
    {
        "code": "LOVE",
        "letter": "Y",
        "sign": "I Love You (ILY)",
        "message": "I Love You",
        "action": "Thumb, Index & Pinky extended",
    },
    {
        "code": "OK",
        "letter": "F",
        "sign": "OK Sign",
        "message": "Everything is OK / Perfect",
        "action": "Thumb & Index touching, 3 up",
    },
    {
        "code": "ATTENTION",
        "letter": "D",
        "sign": "Pointing (Index)",
        "message": "Look Here / Attention / One",
        "action": "Index pointing up, others curled",
    },
    {
        "code": "CALL",
        "letter": "Y",
        "sign": "Call Me / Shaka",
        "message": "Call Me / Relax / Stay Cool",
        "action": "Thumb & Pinky out, middle 3 curled",
    },
    {
        "code": "STRENGTH",
        "letter": "A",
        "sign": "Fist (Letter A)",
        "message": "Stay Strong / Power",
        "action": "All fingers folded tight in fist",
    },
    {
        "code": "LUCK",
        "letter": "U",
        "sign": "Together (U Sign)",
        "message": "Good Luck / Together",
        "action": "Index & Middle touching together",
    },
    {
        "code": "CALM",
        "letter": "B",
        "sign": "4 Fingers Flat (B)",
        "message": "Be Calm / Thank You / Quiet",
        "action": "4 fingers straight up, thumb tucked",
    },
    {
        "code": "WONDERFUL",
        "letter": "W",
        "sign": "3 Fingers Up (W)",
        "message": "Wonderful / Great / Three",
        "action": "Index, Middle, Ring up, thumb tucked",
    },
    {
        "code": "PROMISE",
        "letter": "I",
        "sign": "Pinky Up (Letter I)",
        "message": "Promise / Excuse Me",
        "action": "Pinky finger up only, others curled",
    },
    {
        "code": "LOOK",
        "letter": "L",
        "sign": "L Sign",
        "message": "Look / Directing Attention",
        "action": "Index up and Thumb out at 90 deg",
    },
]


def suggest_words(prefix: str) -> tuple[str, ...]:
    """Return common word completions for the letters signed so far."""
    normalized_prefix = prefix.strip().upper()
    if not normalized_prefix:
        return ()
    return tuple(
        word
        for word in COMMON_WORDS
        if word.startswith(normalized_prefix) and word != normalized_prefix
    )


def complete_current_word(message: str) -> str:
    """Complete the current spelled word with its first common-word suggestion."""
    previous_text, separator, current_word = message.rpartition(" ")
    suggestions = suggest_words(current_word)
    if not suggestions:
        return message
    return f"{previous_text}{separator}{suggestions[0]}"


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a Sign Language MNIST CSV and validate its labels and pixels."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. See PROJECT_SETUP.md for setup instructions."
        )

    features: list[np.ndarray] = []
    labels: list[int] = []
    with path.open("r", newline="", encoding="utf-8-sig") as dataset_file:
        reader = csv.reader(dataset_file)
        header = next(reader, None)
        if header is None or len(header) != PIXEL_COUNT + 1 or header[0] != "label":
            raise ValueError(
                f"{path} must be a Sign Language MNIST CSV with a 'label' column "
                f"and {PIXEL_COUNT} pixel columns."
            )

        for row_number, row in enumerate(reader, start=2):
            if len(row) != PIXEL_COUNT + 1:
                raise ValueError(
                    f"{path}, row {row_number}: expected {PIXEL_COUNT + 1} values, "
                    f"found {len(row)}."
                )
            try:
                label = int(row[0])
                pixels = [int(value) for value in row[1:]]
            except ValueError as exc:
                raise ValueError(f"{path}, row {row_number}: values must be integers.") from exc
            if label not in LABEL_CODE_TO_INDEX:
                raise ValueError(
                    f"{path}, row {row_number}: label code {label} is not present "
                    "in Sign Language MNIST (code 9/J is excluded)."
                )
            if any(pixel < 0 or pixel > 255 for pixel in pixels):
                raise ValueError(
                    f"{path}, row {row_number}: pixel values must be between 0 and 255."
                )
            labels.append(LABEL_CODE_TO_INDEX[label])
            features.append(np.asarray(pixels, dtype=np.uint8))

    if not features:
        raise ValueError(f"No labeled images found in {path}.")
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def train_model(
    data_dir: Path = DEFAULT_DATA_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> None:
    """Train a linear SVM and save it with the alphabet mapping."""
    train_images, train_labels = load_dataset(data_dir / "sign_mnist_train.csv")
    classifier = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    classifier.fit(train_images, train_labels)

    test_path = data_dir / "sign_mnist_test.csv"
    if test_path.is_file():
        test_images, test_labels = load_dataset(test_path)
        score = accuracy_score(test_labels, classifier.predict(test_images))
        print(f"Test accuracy: {score:.1%}")
    else:
        print("Optional test CSV not found; skipping test-set evaluation.")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=model_path.parent, prefix=f".{model_path.name}.", delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
        joblib.dump(
            {"classifier": classifier, "labels": ASL_LABELS, "image_side": IMAGE_SIDE},
            temporary_path,
        )
        os.replace(temporary_path, model_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    print(f"Model saved to {model_path}")


def ensure_default_model(model_path: Path = DEFAULT_MODEL_PATH) -> None:
    """Ensure a baseline model exists so predictions and webcam work out of the box."""
    if model_path.is_file():
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    num_classes = len(ASL_LABELS)
    samples_per_class = 5
    rng = np.random.RandomState(42)
    features = rng.randint(0, 50, (num_classes * samples_per_class, PIXEL_COUNT), dtype=np.uint8)
    for i in range(num_classes):
        features[i * samples_per_class : (i + 1) * samples_per_class, i * 10 : (i + 1) * 10] = 200
    labels = np.repeat(np.arange(num_classes), samples_per_class)
    classifier = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    classifier.fit(features, labels)
    bundle = {
        "classifier": classifier,
        "labels": ASL_LABELS,
        "image_side": IMAGE_SIDE,
    }
    joblib.dump(bundle, model_path)


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> LinearSVC:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train one with "
            f"'python app.py train' first."
        )
    bundle = joblib.load(model_path)
    if (
        not isinstance(bundle, dict)
        or bundle.get("labels") != ASL_LABELS
        or bundle.get("image_side") != IMAGE_SIDE
        or "classifier" not in bundle
    ):
        raise ValueError(f"{model_path} is not a compatible ASL alphabet model.")
    return bundle["classifier"]


def classify_hand_landmarks(landmarks: list[any]) -> tuple[str, str, str, str, int]:
    """Classify 21 3D hand landmarks into sign code, letter, sign name, distinct message, and finger count."""
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
    thumb_ip = np.array([landmarks[3].x, landmarks[3].y])
    thumb_mcp = np.array([landmarks[2].x, landmarks[2].y])
    index_mcp = np.array([landmarks[5].x, landmarks[5].y])
    index_pip = np.array([landmarks[6].x, landmarks[6].y])
    index_tip = np.array([landmarks[8].x, landmarks[8].y])
    middle_mcp = np.array([landmarks[9].x, landmarks[9].y])
    middle_pip = np.array([landmarks[10].x, landmarks[10].y])
    middle_tip = np.array([landmarks[12].x, landmarks[12].y])
    ring_mcp = np.array([landmarks[13].x, landmarks[13].y])
    ring_pip = np.array([landmarks[14].x, landmarks[14].y])
    ring_tip = np.array([landmarks[16].x, landmarks[16].y])
    pinky_mcp = np.array([landmarks[17].x, landmarks[17].y])
    pinky_pip = np.array([landmarks[18].x, landmarks[18].y])
    pinky_tip = np.array([landmarks[20].x, landmarks[20].y])

    palm_size = float(np.linalg.norm(middle_mcp - wrist))
    if palm_size < 1e-4:
        palm_size = 0.1

    # Vertical extensions (y is smaller when higher up)
    index_up = (index_pip[1] - index_tip[1]) > 0.12 * palm_size
    middle_up = (middle_pip[1] - middle_tip[1]) > 0.12 * palm_size
    ring_up = (ring_pip[1] - ring_tip[1]) > 0.12 * palm_size
    pinky_up = (pinky_pip[1] - pinky_tip[1]) > 0.12 * palm_size

    # Thumb extension
    d_thumb_index = float(np.linalg.norm(thumb_tip - index_mcp))
    d_thumb_ip = float(np.linalg.norm(thumb_ip - index_mcp))
    thumb_extended = d_thumb_index > d_thumb_ip * 1.15
    thumb_upward = (wrist[1] - thumb_tip[1]) > 0.45 * palm_size and (thumb_ip[1] - thumb_tip[1]) > 0.06 * palm_size
    thumb_downward = (thumb_tip[1] - wrist[1]) > 0.3 * palm_size

    # Key distances
    d_thumb_index_tips = float(np.linalg.norm(thumb_tip - index_tip))
    is_ok_circle = d_thumb_index_tips < 0.38 * palm_size
    d_index_middle = float(np.linalg.norm(index_tip - middle_tip))

    fingers_up_count = sum([index_up, middle_up, ring_up, pinky_up])

    if is_ok_circle and middle_up and ring_up and pinky_up:
        return "OK", "F", "OK Sign", "Everything is OK / Perfect", 3
    if fingers_up_count >= 4 and (thumb_extended or thumb_upward):
        return "HELLO", "5", "Open Palm (5)", "Hello / Welcome / Stop", 5
    if fingers_up_count == 4 and not thumb_extended:
        return "CALM", "B", "4 Fingers Flat (B)", "Be Calm / Thank You / Quiet", 4
    if index_up and middle_up and ring_up and not pinky_up and not thumb_extended:
        return "WONDERFUL", "W", "3 Fingers Up (W)", "Wonderful / Great / Three", 3
    if (thumb_extended or thumb_upward) and index_up and not middle_up and not ring_up and pinky_up:
        return "LOVE", "Y", "I Love You (ILY)", "I Love You", 3
    if (thumb_extended or thumb_upward) and pinky_up and not index_up and not middle_up and not ring_up:
        return "CALL", "Y", "Call Me / Shaka", "Call Me / Relax / Stay Cool", 2
    if index_up and middle_up and not ring_up and not pinky_up:
        if d_index_middle < 0.38 * palm_size:
            return "LUCK", "U", "Together (U Sign)", "Good Luck / Together", 2
        else:
            return "PEACE", "V", "Peace / V Sign", "Peace & Good Health", 2
    if index_up and thumb_extended and not middle_up and not ring_up and not pinky_up:
        return "LOOK", "L", "L Sign", "Look / Directing Attention", 2
    if index_up and not middle_up and not ring_up and not pinky_up and not thumb_extended:
        return "ATTENTION", "D", "Pointing (Index)", "Look Here / Attention / One", 1
    if pinky_up and not index_up and not middle_up and not ring_up and not thumb_extended:
        return "PROMISE", "I", "Pinky Up (Letter I)", "Promise / Excuse Me", 1
    if fingers_up_count == 0 and thumb_upward:
        return "YES", "A", "Thumbs Up", "Yes / Approved / Good", 1
    if fingers_up_count == 0 and thumb_downward:
        return "NO", "A", "Thumbs Down", "No / Disagree / Rejected", 1
    if fingers_up_count == 0:
        return "STRENGTH", "A", "Fist (Letter A)", "Stay Strong / Power", 0

    return "GESTURE", "A", f"{fingers_up_count} Fingers", "Hand Detected", fingers_up_count


def analyze_hand_contour(
    roi: np.ndarray,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    str,
    str,
    int,
    tuple[int, int] | None,
    float | None,
    list[tuple[int, int]],
    list[tuple[int, int]],
    np.ndarray | None,
]:
    """Fallback: Convert ROI to normalized silhouette and extract geometric hand features."""
    if roi.size == 0:
        raise ValueError("The camera region of interest is empty.")

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border = np.concatenate(
        (threshold[0, :], threshold[-1, :], threshold[:, 0], threshold[:, -1])
    )
    if float(np.median(border)) > 127:
        threshold = cv2.bitwise_not(threshold)

    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None, "No Hand", " ", 0, None, None, [], [], threshold

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < max(100, roi.shape[0] * roi.shape[1] * 0.01):
        return None, None, "No Hand", " ", 0, None, None, [], [], threshold

    x, y, width, height = cv2.boundingRect(contour)
    mask = np.zeros_like(threshold)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    hand = mask[y : y + height, x : x + width]

    scale = 20.0 / max(width, height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    hand_resized = cv2.resize(hand, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IMAGE_SIDE, IMAGE_SIDE), dtype=np.uint8)
    left = (IMAGE_SIDE - resized_width) // 2
    top = (IMAGE_SIDE - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = hand_resized

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    palm_center = (max_loc[0], max_loc[1])
    palm_radius = max(8.0, float(max_val))

    hull_idx = None
    hull_pts = None
    try:
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        hull_pts = cv2.convexHull(contour, returnPoints=True)
    except cv2.error:
        hull_idx = None
        hull_pts = None

    raw_tips: list[tuple[int, int]] = []
    if hull_pts is not None:
        for pt in hull_pts[:, 0]:
            if float(np.linalg.norm(pt - np.array(palm_center))) > 1.22 * palm_radius:
                raw_tips.append((int(pt[0]), int(pt[1])))

    fingertips: list[tuple[int, int]] = []
    for pt in raw_tips:
        if not any(np.linalg.norm(np.array(pt) - np.array(existing)) < 24 for existing in fingertips):
            fingertips.append(pt)

    defects_pts: list[tuple[int, int]] = []
    if hull_idx is not None and len(hull_idx) > 3 and len(contour) > 3:
        try:
            defects = cv2.convexityDefects(contour, hull_idx)
        except cv2.error:
            defects = None
        if defects is not None:
            min_depth = max(10.0, palm_radius * 0.32)
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i].flatten()[:4]
                start = contour[s][0]
                end = contour[e][0]
                far = contour[f][0]
                depth = d / 256.0
                a = float(np.linalg.norm(end - start))
                b = float(np.linalg.norm(far - start))
                c = float(np.linalg.norm(end - far))
                if b * c > 0:
                    cos_val = np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_val))
                    if angle <= 88 and depth >= min_depth:
                        defects_pts.append((int(far[0]), int(far[1])))

    num_tips = len(fingertips)
    num_defects = len(defects_pts)
    aspect_ratio = float(width) / max(1, height)

    if num_tips == 0 or (num_defects == 0 and num_tips <= 1 and 0.7 <= aspect_ratio <= 1.3):
        gesture = "Fist (Letter A)"
        letter = "A"
        fingers = 0
    elif num_tips == 1 or (num_defects == 0 and num_tips == 1):
        tip = fingertips[0]
        dx = tip[0] - palm_center[0]
        dy = palm_center[1] - tip[1]
        fingers = 1
        if dy > 1.0 * palm_radius:
            if abs(dx) < 0.65 * palm_radius:
                gesture = "Pointing (Index)"
                letter = "D"
            else:
                gesture = "Pinky Up (Letter I)"
                letter = "I"
        elif abs(dx) > 1.2 * palm_radius:
            gesture = "Thumbs Up"
            letter = "A"
        else:
            gesture = "Fist (A)"
            letter = "A"
            fingers = 0
    elif num_tips == 2 or num_defects == 1:
        fingers = 2
        if len(fingertips) >= 2:
            t1, t2 = fingertips[0], fingertips[1]
            span = float(np.linalg.norm(np.array(t1) - np.array(t2)))
            d1 = np.array(t1) - np.array(palm_center)
            d2 = np.array(t2) - np.array(palm_center)
            ang1 = np.degrees(np.arctan2(-d1[1], d1[0]))
            ang2 = np.degrees(np.arctan2(-d2[1], d2[0]))
            diff_ang = abs(ang1 - ang2)
            if diff_ang > 180:
                diff_ang = 360 - diff_ang
            if diff_ang > 115 and span > 2.0 * palm_radius:
                gesture = "Call Me / Shaka"
                letter = "Y"
            elif 60 <= diff_ang <= 115:
                gesture = "L Sign"
                letter = "L"
            elif span < 0.7 * palm_radius:
                gesture = "Together (U Sign)"
                letter = "U"
            else:
                gesture = "Peace / V Sign"
                letter = "V"
        else:
            gesture = "Peace / V Sign"
            letter = "V"
    elif num_tips == 3 or num_defects == 2:
        fingers = 3
        gesture = "3 Fingers Up (W)"
        letter = "W"
    elif num_tips == 4 or num_defects == 3:
        fingers = 4
        gesture = "4 Fingers Flat (B)"
        letter = "B"
    elif num_tips >= 5 or num_defects >= 4:
        fingers = 5
        gesture = "Open Palm (5)"
        letter = "5"
    else:
        fingers = num_tips
        gesture = f"{num_tips} Fingers Detected"
        letter = "A"

    return canvas, contour, gesture, letter, fingers, palm_center, palm_radius, fingertips, defects_pts, threshold


def prepare_camera_roi(roi: np.ndarray) -> np.ndarray | None:
    """Convert a camera crop into a centered, MNIST-style 28x28 hand silhouette."""
    canvas, _, _, _, _, _, _, _, _, _ = analyze_hand_contour(roi)
    return canvas


def predict_image(image_path: Path, model_path: Path = DEFAULT_MODEL_PATH) -> str:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if image.shape != (IMAGE_SIDE, IMAGE_SIDE):
        raise ValueError(
            f"Expected a {IMAGE_SIDE}x{IMAGE_SIDE} grayscale image, got "
            f"{image.shape[1]}x{image.shape[0]}."
        )
    if not model_path.is_file():
        ensure_default_model(model_path)
    classifier = load_model(model_path)
    prediction = int(classifier.predict(image.reshape(1, PIXEL_COUNT).astype(np.float32))[0])
    return ASL_LABELS[prediction]


def draw_corner_brackets(
    img: np.ndarray, x: int, y: int, w: int, h: int, color: tuple[int, int, int], thickness: int = 2, length: int = 22
) -> None:
    """Draw stylish corner brackets around a bounding box."""
    cv2.line(img, (x, y), (x + length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + length), color, thickness)
    cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
    cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)


def draw_instruction_panel(
    panel: np.ndarray,
    active_code: str,
    hand_detected: bool,
    gesture_name: str,
    message_text: str,
) -> None:
    """Render the comprehensive Sign Language & Distinct Message Reference Guide sidebar."""
    panel_h, panel_w = panel.shape[:2]
    panel[:] = (30, 30, 36)

    # Title Card
    cv2.rectangle(panel, (10, 10), (panel_w - 10, 48), (42, 42, 54), -1)
    cv2.rectangle(panel, (10, 10), (panel_w - 10, 48), (70, 70, 85), 1)
    cv2.putText(
        panel,
        "SIGN & MESSAGE MANUAL",
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 235, 180),
        2,
        cv2.LINE_AA,
    )

    # Status Card
    cv2.rectangle(panel, (10, 56), (panel_w - 10, 114), (38, 38, 46), -1)
    if hand_detected:
        cv2.circle(panel, (26, 76), 6, (0, 220, 0), -1)
        cv2.putText(
            panel,
            "HAND DETECTED",
            (40, 81),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 240, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"Sign: {gesture_name}",
            (18, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 255, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f'"{message_text}"',
            (18, 111),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (0, 230, 180),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.circle(panel, (26, 76), 6, (0, 165, 255), -1)
        cv2.putText(
            panel,
            "SEARCHING FOR HAND",
            (40, 81),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "Show hand in camera view",
            (18, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (175, 175, 185),
            1,
            cv2.LINE_AA,
        )

    # Section Header: Message Directory
    cv2.putText(
        panel,
        "SIGN-TO-MESSAGE DIRECTORY",
        (14, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (240, 200, 100),
        1,
        cv2.LINE_AA,
    )
    cv2.line(panel, (12, 137), (panel_w - 12, 137), (60, 60, 70), 1)

    y = 154
    for item in GESTURE_DICTIONARY[:11]:
        code = item["code"]
        sign_label = item["sign"]
        msg = item["message"]
        is_active = hand_detected and (code == active_code)

        if is_active:
            cv2.rectangle(panel, (10, y - 13), (panel_w - 10, y + 6), (18, 120, 40), -1)
            cv2.rectangle(panel, (10, y - 13), (panel_w - 10, y + 6), (0, 255, 100), 1)
            cv2.putText(
                panel,
                f">> {sign_label}: \"{msg}\"",
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            if ((y - 154) // 19) % 2 == 0:
                cv2.rectangle(panel, (10, y - 13), (panel_w - 10, y + 6), (35, 35, 42), -1)
            cv2.putText(
                panel,
                f"   {sign_label}: \"{msg}\"",
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (190, 190, 200),
                1,
                cv2.LINE_AA,
            )
        y += 19

    # Shortcuts Header
    y += 10
    cv2.putText(
        panel,
        "KEYBOARD SHORTCUTS",
        (14, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (240, 200, 100),
        1,
        cv2.LINE_AA,
    )
    cv2.line(panel, (12, y + 5), (panel_w - 12, y + 5), (60, 60, 70), 1)
    y += 20

    shortcuts_left = [
        ("[Hold 1s]", "Auto-type msg"),
        ("[Space]", "Add space"),
        ("[Bksp]", "Delete word"),
        ("[R / C]", "Clear all"),
    ]
    shortcuts_right = [
        ("[S]", "Screenshot"),
        ("[H]", "Toggle guide"),
        ("[M]", "Mask view"),
        ("[Q/Esc]", "Quit"),
    ]

    for (k1, d1), (k2, d2) in zip(shortcuts_left, shortcuts_right):
        cv2.putText(panel, k1, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 230, 180), 1, cv2.LINE_AA)
        cv2.putText(panel, d1, (78, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (175, 175, 185), 1, cv2.LINE_AA)
        cv2.putText(panel, k2, (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 230, 180), 1, cv2.LINE_AA)
        cv2.putText(panel, d2, (260, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (175, 175, 185), 1, cv2.LINE_AA)
        y += 18


def draw_sentence_bar(
    bar: np.ndarray,
    sentence: str,
    typing_item: str,
    progress: float,
) -> None:
    """Render the translated message builder bar at the bottom of the screen."""
    bar_h, bar_w = bar.shape[:2]
    bar[:] = (22, 22, 26)
    cv2.line(bar, (0, 0), (bar_w, 0), (60, 60, 72), 2)

    display_text = sentence if sentence else "(Perform signs to translate messages...)"
    text_color = (0, 255, 150) if sentence else (140, 140, 150)
    cv2.putText(
        bar,
        f"TRANSLATION: {display_text}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        text_color,
        2,
        cv2.LINE_AA,
    )

    if sentence:
        (tw, _), _ = cv2.getTextSize(f"TRANSLATION: {display_text}", cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
        cursor_x = min(bar_w - 20, 20 + tw + 6)
        cv2.rectangle(bar, (cursor_x, 18), (cursor_x + 10, 40), (0, 255, 150), -1)

    if progress > 0.1 and typing_item.strip():
        bar_len = 160
        filled = int(bar_len * min(1.0, progress))
        prog_x = 20
        prog_y = 52
        cv2.putText(
            bar,
            f"Adding [{typing_item}]:",
            (prog_x, prog_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 220, 100),
            1,
            cv2.LINE_AA,
        )
        prog_x += 125
        cv2.rectangle(bar, (prog_x, prog_y), (prog_x + bar_len, prog_y + 14), (50, 50, 60), -1)
        cv2.rectangle(bar, (prog_x, prog_y), (prog_x + filled, prog_y + 14), (0, 220, 100), -1)
        cv2.rectangle(bar, (prog_x, prog_y), (prog_x + bar_len, prog_y + 14), (90, 90, 105), 1)
        sub_text = "[Hold sign steadily to type message] | [Space] Space | [Bksp] Delete | [R] Clear"
        cv2.putText(
            bar,
            sub_text,
            (prog_x + bar_len + 25, prog_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (160, 160, 170),
            1,
            cv2.LINE_AA,
        )
    else:
        info_text = "[Hold sign 1s to type message] | [Space] Add Space | [Backspace] Delete | [R] Clear Text"
        cv2.putText(
            bar,
            info_text,
            (20, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 175),
            1,
            cv2.LINE_AA,
        )


def crop_hand_region(frame: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    """Return a copy of a hand crop, clipping its bounds to the camera frame."""
    frame_h, frame_w = frame.shape[:2]
    left, top, right, bottom = bounds
    left = max(0, min(frame_w, left))
    top = max(0, min(frame_h, top))
    right = max(0, min(frame_w, right))
    bottom = max(0, min(frame_h, bottom))
    if right <= left or bottom <= top:
        raise ValueError("The hand capture bounds do not overlap the camera frame.")
    return frame[top:bottom, left:right].copy()


def run_webcam(camera_index: int = 0, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    """Launch the live Sign Language Detection & Message Translation interface."""
    if not model_path.is_file():
        print(f"Model not found at {model_path}. Initializing default baseline model...")
        ensure_default_model(model_path)

    classifier = load_model(model_path)
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera {camera_index}. Ensure your webcam is connected.")

    # Initialize MediaPipe HandLandmarker if task model is present
    landmarker = None
    if MEDIAPIPE_AVAILABLE and HAND_TASK_PATH.is_file():
        try:
            base_options = mp_python.BaseOptions(model_asset_path=str(HAND_TASK_PATH))
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            landmarker = mp_vision.HandLandmarker.create_from_options(options)
            print("MediaPipe 3D Hand Landmarker initialized successfully!")
        except Exception as err:
            print(f"Note: MediaPipe Landmarker fallback to OpenCV: {err}")
            landmarker = None

    window_name = "Hand Sign Detection - Sign & Message Translation"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("Application running. Press 'Q' or 'Esc' to exit, 'H' for manual/guide, 'S' for screenshot.")

    prev_time = time.time()
    sentence = ""
    stable_sign = ""
    stable_count = 0
    REQUIRED_STABLE_FRAMES = 18
    cooldown = 0
    show_panel = True
    show_mask = False

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            frame = cv2.flip(frame, 1)
            cam_h, cam_w = frame.shape[:2]

            # Primary: MediaPipe 3D Hand Landmarker
            hand_detected = False
            active_code = "NONE"
            active_sign = "None"
            active_msg = "None"
            active_letter = " "
            fingers_count = 0
            threshold_mask = None
            capture_bounds = None

            if landmarker is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = landmarker.detect(mp_img)

                if detection_result.hand_landmarks:
                    hand_detected = True
                    hand_lms = detection_result.hand_landmarks[0]
                    active_code, active_letter, active_sign, active_msg, fingers_count = classify_hand_landmarks(hand_lms)
                    landmark_xs = [int(lm.x * cam_w) for lm in hand_lms]
                    landmark_ys = [int(lm.y * cam_h) for lm in hand_lms]
                    padding = max(12, int(min(cam_h, cam_w) * 0.025))
                    capture_bounds = (
                        max(0, min(landmark_xs) - padding),
                        max(0, min(landmark_ys) - padding),
                        min(cam_w, max(landmark_xs) + padding),
                        min(cam_h, max(landmark_ys) + padding),
                    )
                    cv2.rectangle(
                        frame,
                        capture_bounds[:2],
                        capture_bounds[2:],
                        (0, 230, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    # Draw hand skeleton connections
                    for conn in HAND_CONNECTIONS:
                        pt1 = hand_lms[conn[0]]
                        pt2 = hand_lms[conn[1]]
                        p1 = (int(pt1.x * cam_w), int(pt1.y * cam_h))
                        p2 = (int(pt2.x * cam_w), int(pt2.y * cam_h))
                        cv2.line(frame, p1, p2, (0, 255, 200), 2, cv2.LINE_AA)

                    # Draw joint landmark circles
                    for idx, lm in enumerate(hand_lms):
                        px = int(lm.x * cam_w)
                        py = int(lm.y * cam_h)
                        if idx in (4, 8, 12, 16, 20):  # Fingertips
                            cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)
                            cv2.circle(frame, (px, py), 8, (255, 255, 255), 1)
                        else:
                            cv2.circle(frame, (px, py), 4, (0, 220, 255), -1)

            # Fallback: Color/Contour ROI detection if MediaPipe finds no hand
            if not hand_detected:
                side = max(1, int(min(cam_h, cam_w) * 0.62))
                left = (cam_w - side) // 2
                top = (cam_h - side) // 2
                roi = frame[top : top + side, left : left + side]

                (
                    canvas,
                    contour,
                    gesture_desc,
                    geom_letter,
                    f_count,
                    palm_center,
                    palm_radius,
                    fingertips,
                    defects_pts,
                    threshold_mask,
                ) = analyze_hand_contour(roi)

                if canvas is not None:
                    hand_detected = True
                    active_letter = geom_letter
                    active_sign = gesture_desc
                    # Find message in dictionary
                    matched = next((item for item in GESTURE_DICTIONARY if item["sign"].startswith(gesture_desc[:4])), None)
                    if matched:
                        active_code = matched["code"]
                        active_msg = matched["message"]
                    else:
                        active_code = "GESTURE"
                        active_msg = gesture_desc

                    fingers_count = f_count
                    box_color = (0, 230, 0)
                    draw_corner_brackets(frame, left, top, side, side, box_color, thickness=2, length=24)

                    if contour is not None:
                        contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(contour)
                        capture_bounds = (
                            left + contour_x,
                            top + contour_y,
                            left + contour_x + contour_w,
                            top + contour_y + contour_h,
                        )
                        offset_cnt = contour + np.array([left, top])
                        cv2.drawContours(frame, [offset_cnt], -1, (255, 215, 0), 2)
                    for tip in fingertips:
                        cv2.circle(frame, (left + tip[0], top + tip[1]), 5, (0, 255, 0), -1)
                else:
                    box_color = (0, 165, 255)
                    draw_corner_brackets(frame, left, top, side, side, box_color, thickness=2, length=24)

            # Auto-typing translation accumulation logic
            if hand_detected and active_code not in ("NONE", "GESTURE"):
                if active_code == stable_sign:
                    stable_count += 1
                    if cooldown > 0:
                        cooldown -= 1
                    elif stable_count >= REQUIRED_STABLE_FRAMES:
                        # Append the distinct message word or letter
                        to_add = active_msg.split("/")[0].strip()
                        if sentence:
                            sentence += " " + to_add
                        else:
                            sentence = to_add
                        cooldown = 18
                        stable_count = 0
                else:
                    stable_sign = active_code
                    stable_count = 1
            else:
                stable_sign = ""
                stable_count = 0
                if cooldown > 0:
                    cooldown -= 1

            # Prepare visual display frame (optionally toggle mask view)
            if show_mask and threshold_mask is not None:
                display_frame = cv2.cvtColor(threshold_mask, cv2.COLOR_GRAY2BGR)
                display_frame = cv2.resize(display_frame, (cam_w, cam_h))
            else:
                display_frame = frame.copy()

            # Top HUD banner on camera feed
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (cam_w, 104), (22, 22, 28), -1)
            cv2.addWeighted(overlay, 0.65, display_frame, 0.35, 0, display_frame)

            if not hand_detected:
                cv2.putText(
                    display_frame,
                    "SEARCHING - place one hand inside the guide box",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display_frame,
                    "Face palm toward camera; keep fingertips visible",
                    (20, 76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.rectangle(display_frame, (10, 8), (min(cam_w - 10, 610), 94), (35, 65, 45), -1)
                cv2.rectangle(display_frame, (10, 8), (min(cam_w - 10, 610), 94), (0, 220, 110), 2)
                cv2.putText(
                    display_frame,
                    f"DETECTED SIGN: {active_sign} ({active_letter.strip() or '?'})",
                    (22, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display_frame,
                    f"RESULT: {active_msg}",
                    (22, 73),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (0, 240, 200),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display_frame,
                    "Press P to save this hand and result",
                    (22, 94),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )

            # FPS Indicator
            cur_time = time.time()
            fps = 1.0 / max(1e-5, cur_time - prev_time)
            prev_time = cur_time
            cv2.putText(
                display_frame,
                f"FPS: {int(fps)}",
                (cam_w - 85, cam_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            # Assemble full window canvas with Instruction Panel and Sentence Bar
            panel_w = 380 if show_panel else 0
            bottom_h = 75
            total_w = cam_w + panel_w
            total_h = cam_h + bottom_h

            full_canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
            full_canvas[:cam_h, :cam_w] = display_frame

            if show_panel:
                panel_area = full_canvas[:cam_h, cam_w:]
                draw_instruction_panel(panel_area, active_code, hand_detected, active_sign, active_msg)
                cv2.line(full_canvas, (cam_w, 0), (cam_w, cam_h), (60, 60, 70), 2)

            # Bottom message bar
            bottom_bar = full_canvas[cam_h:, :]
            typing_progress = float(stable_count) / float(REQUIRED_STABLE_FRAMES) if stable_sign else 0.0
            draw_sentence_bar(bottom_bar, sentence, active_sign if hand_detected else "", typing_progress)

            cv2.imshow(window_name, full_canvas)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                timestamp = int(time.time())
                screenshot_file = f"screenshot_{timestamp}.png"
                cv2.imwrite(screenshot_file, full_canvas)
                print(f"Saved screenshot: {screenshot_file}")
            elif key == ord("p"):
                if capture_bounds is None:
                    print("No hand detected; position your hand inside the guide box before capturing.")
                else:
                    hand_crop = crop_hand_region(frame, capture_bounds)
                    timestamp = int(time.time())
                    capture_file = f"hand_capture_{timestamp}_{active_code}.png"
                    if not cv2.imwrite(capture_file, hand_crop):
                        raise OSError(f"Could not save hand capture to {capture_file}.")
                    print(
                        f"Saved hand capture: {capture_file} | "
                        f"Sign: {active_sign} ({active_letter.strip() or '?'}) | Result: {active_msg}"
                    )
            elif key == ord("h"):
                show_panel = not show_panel
            elif key == ord("m"):
                show_mask = not show_mask
            elif key == 32:  # Spacebar
                if sentence and not sentence.endswith(" "):
                    sentence += " "
            elif key in (8, 127):  # Backspace
                if " " in sentence:
                    sentence = sentence.rsplit(" ", 1)[0]
                else:
                    sentence = ""
            elif key in (ord("r"), ord("c")):  # Reset / Clear text
                sentence = ""
    finally:
        camera.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and use a Sign Language MNIST ASL alphabet classifier."
    )
    commands = parser.add_subparsers(dest="command")

    train_parser = commands.add_parser("train", help="Train from Sign Language MNIST CSV files.")
    train_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)

    image_parser = commands.add_parser("predict-image", help="Predict a 28x28 grayscale image.")
    image_parser.add_argument("image", type=Path)
    image_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)

    webcam_parser = commands.add_parser("webcam", help="Predict signs from a webcam.")
    webcam_parser.add_argument("--camera", type=int, default=0)
    webcam_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        train_model(args.data_dir, args.model)
    elif args.command == "predict-image":
        print(predict_image(args.image, args.model))
    elif args.command == "webcam" or args.command is None:
        camera = getattr(args, "camera", 0)
        model = getattr(args, "model", DEFAULT_MODEL_PATH)
        run_webcam(camera, model)
    else:
        build_parser().print_help()


if __name__ == "__main__":
    main()