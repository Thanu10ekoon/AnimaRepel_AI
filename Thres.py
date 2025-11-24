import os
import time
import csv
import threading
from datetime import datetime
import platform
import subprocess
import json
import argparse

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------- CONFIG --------------------
MODEL_PATH = "object_identifier_model.tflite"   # TFLite model exported from Colab
BUZZER_SOUND = "buzzer.wav"                     # Path to buzzer sound file
LOG_PATH = "buzzer_log.csv"                     # CSV file to store buzzer events

# Optional: path to the folder with your training images
# (subfolders per class, same as you used for training in Colab)
DATA_DIR = "AnimaRepel"                         # If you copied dataset; otherwise JSON is enough

# JSON file exported from Colab with train_generator.class_indices
CLASS_INDICES_JSON = "class_indices.json"

# Per-animal threshold config file
THRESHOLDS_JSON = "thresholds.json"

IMG_SIZE = 80                                   # Must match training
DEFAULT_CONFIDENCE_THRESHOLD = 0.5              # Fallback if not overridden in thresholds.json
DANGER_KEYWORD = "dang"                         # Any class name containing this is treated as "harmful"
BEEP_MIN_INTERVAL = 1.0                         # Seconds between beeps (to avoid spamming)


# -------------------- MODEL + CLASSES --------------------


def load_tflite_model(model_path: str):
    """
    Load a TFLite model using tf.lite.Interpreter.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"[INFO] Loading TFLite model from {model_path} ...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print("[INFO] TFLite model loaded.")
    print(f"[INFO] Input details: shape={input_details['shape']}, dtype={input_details['dtype']}")
    print(f"[INFO] Output details: shape={output_details['shape']}, dtype={output_details['dtype']}")
    return interpreter, input_details, output_details


def load_class_mapping_from_json(json_path: str, num_classes: int):
    """
    Load index->class mapping from class_indices.json (exported from Colab).
    JSON should be of the form: { "WildBoar_dang": 0, "Deer_ndan": 1, ... }
    """
    if not os.path.exists(json_path):
        return None

    with open(json_path, "r") as f:
        class_indices = json.load(f)  # name -> index
    index_to_class = {}

    for name, idx in class_indices.items():
        index_to_class[int(idx)] = name

    # Sanity: make sure all indices from 0..num_classes-1 are covered
    missing = [i for i in range(num_classes) if i not in index_to_class]
    if missing:
        print(f"[WARN] JSON mapping missing indices {missing}; they will get generic names.")
        for i in missing:
            index_to_class[i] = f"class_{i}"

    print("[INFO] Loaded class mapping from JSON:", index_to_class)
    return index_to_class


def build_class_mapping_from_directory(data_dir: str, num_classes: int):
    """
    Try to rebuild the index->class mapping via flow_from_directory.
    Only used if JSON is not available.
    """
    if not os.path.isdir(data_dir):
        return None

    print(f"[INFO] Building class mapping from directory: {data_dir}")
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        class_mode="categorical",
        shuffle=False,
    )
    class_indices = generator.class_indices  # dict: class_name -> index
    index_to_class = {v: k for k, v in class_indices.items()}
    print(f"[INFO] Classes found from directory: {index_to_class}")
    return index_to_class


def get_index_to_class_mapping(num_classes: int):
    """
    Try JSON first, then directory; fall back to generic names.
    """
    # 1) JSON (best: exactly matches training generator)
    index_to_class = load_class_mapping_from_json(CLASS_INDICES_JSON, num_classes)
    if index_to_class is not None:
        return index_to_class

    # 2) Directory (if you copied dataset locally)
    index_to_class = build_class_mapping_from_directory(DATA_DIR, num_classes)
    if index_to_class is not None:
        return index_to_class

    # 3) Fallback: generic names
    index_to_class = {i: f"class_{i}" for i in range(num_classes)}
    print("[WARN] No JSON or DATA_DIR found. Using generic class names:", index_to_class)
    return index_to_class


def ensure_log_file(log_path: str):
    """
    Ensure the CSV log file exists and has a header.
    """
    file_exists = os.path.exists(log_path)
    if not file_exists:
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "animal", "confidence_percent", "raw_class_name"])
        print(f"[INFO] Created log file: {log_path}")
    else:
        print(f"[INFO] Using existing log file: {log_path}")


def log_buzzer_event(log_path: str, animal: str, confidence: float, raw_class_name: str):
    """
    Append a buzzer event to the CSV log file.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, animal, f"{confidence * 100:.2f}", raw_class_name])
    print(f"[LOG] {timestamp} | {animal} | {confidence*100:.2f}%")


def play_buzzer_async():
    """
    Play the buzzer sound in a separate thread so the main loop is not blocked.
    Uses:
      - winsound on Windows
      - ffplay (from ffmpeg) on Linux/WSL
    """
    if not os.path.exists(BUZZER_SOUND):
        print(f"[WARN] Buzzer sound file not found: {BUZZER_SOUND}")
        return

    def _play():
        try:
            system = platform.system()
            if system == "Windows":
                # Native Windows
                import winsound
                winsound.PlaySound(BUZZER_SOUND, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                # Linux / WSL: use ffplay (from ffmpeg)
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", BUZZER_SOUND],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as e:
            print(f"[ERROR] Could not play buzzer: {e}")

    t = threading.Thread(target=_play, daemon=True)
    t.start()


def pretty_animal_name(class_name: str) -> str:
    """
    Convert class name like 'WildBoar_dang' or 'WildBoar_ndan' into 'WildBoar'
    for nicer logging & threshold lookup. If no underscore, return as-is.
    """
    parts = class_name.split("_")
    if len(parts) > 1:
        return "_".join(parts[:-1])
    return class_name


def preprocess_frame(frame, input_details):
    """
    Convert a BGR frame (from OpenCV) into a normalized tensor matching the TFLite input.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    arr = resized.astype("float32") / 255.0

    # Ensure shape matches the model's expected input
    if len(input_details["shape"]) == 4:
        arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)

    # Cast to correct dtype
    arr = arr.astype(input_details["dtype"])
    return arr


# -------------------- THRESHOLDS CONFIG --------------------


def load_thresholds_config(json_path: str, default_fallback: float, base_animals=None):
    """
    Load per-animal thresholds from thresholds.json and ensure all base_animals exist.

    Expected format:

    {
      "default_threshold": 0.5,
      "per_animal": {
        "WildBoar": 0.6,
        "Elephant": 0.7
      }
    }
    """
    default_threshold = default_fallback
    per_animal = {}
    changed = False

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            if "default_threshold" in data:
                default_threshold = float(data["default_threshold"])
            if "per_animal" in data and isinstance(data["per_animal"], dict):
                for k, v in data["per_animal"].items():
                    per_animal[str(k)] = float(v)

            print(f"[INFO] Loaded thresholds from {json_path}")
        except Exception as e:
            print(f"[WARN] Failed to load {json_path}: {e}")
            print(f"       Falling back to global threshold = {default_threshold}")
            per_animal = {}
            changed = True
    else:
        print(f"[INFO] No {json_path} found. It will be created.")

    # Ensure all animals have entries
    if base_animals is not None:
        for animal in base_animals:
            if animal not in per_animal:
                per_animal[animal] = default_threshold
                changed = True

    # Save back if file missing or we added animals or had errors
    if changed or not os.path.exists(json_path):
        data = {
            "default_threshold": default_threshold,
            "per_animal": per_animal,
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] thresholds.json updated/created with current animals.")

    print(f"[INFO] Threshold config: default={default_threshold}, per_animal={per_animal}")
    return default_threshold, per_animal


def is_dangerous_class(class_name: str, confidence: float, threshold: float) -> bool:
    """
    Decide if a predicted class is harmful enough to trigger the buzzer.
    We treat any class whose name contains 'dang' (case-insensitive)
    and has confidence above the GIVEN threshold as dangerous.
    """
    return (DANGER_KEYWORD in class_name.lower()) and (confidence >= threshold)


# -------------------- THRESHOLD EDITOR MODE --------------------


def load_base_animals_from_class_indices():
    """
    Read class_indices.json and derive base animal names.
    """
    if not os.path.exists(CLASS_INDICES_JSON):
        raise FileNotFoundError(
            f"{CLASS_INDICES_JSON} not found. Cannot derive animals for thresholds."
        )

    with open(CLASS_INDICES_JSON, "r") as f:
        class_indices = json.load(f)  # name -> index

    base_animals = sorted({pretty_animal_name(name) for name in class_indices.keys()})
    return base_animals


def run_threshold_editor():
    """
    Interactive CLI to edit per-animal thresholds and save to thresholds.json.
    """
    print("=== Threshold Editor Mode ===")
    base_animals = load_base_animals_from_class_indices()
    default_threshold, per_animal = load_thresholds_config(
        THRESHOLDS_JSON, DEFAULT_CONFIDENCE_THRESHOLD, base_animals
    )

    while True:
        print("\nCurrent thresholds:")
        for i, animal in enumerate(base_animals):
            thr = per_animal.get(animal, default_threshold)
            print(f"  [{i}] {animal}: {thr:.3f}")
        print(f"\nDefault threshold: {default_threshold:.3f}")
        print("\nOptions:")
        print("  [number] - edit that animal's threshold")
        print("  d        - edit default threshold")
        print("  q        - save & quit")

        choice = input("Enter choice: ").strip().lower()

        if choice == "q":
            break
        elif choice == "d":
            new_thr_str = input(f"New default threshold (0-1, current {default_threshold:.3f}): ").strip()
            if new_thr_str:
                try:
                    new_thr = float(new_thr_str)
                    if 0.0 <= new_thr <= 1.0:
                        default_threshold = new_thr
                        print(f"[INFO] Default threshold updated to {default_threshold:.3f}")
                    else:
                        print("[WARN] Threshold should be between 0 and 1.")
                except ValueError:
                    print("[WARN] Invalid number.")
        else:
            try:
                idx = int(choice)
                if 0 <= idx < len(base_animals):
                    animal = base_animals[idx]
                    current_thr = per_animal.get(animal, default_threshold)
                    new_thr_str = input(
                        f"New threshold for {animal} (0-1, current {current_thr:.3f}): "
                    ).strip()
                    if new_thr_str:
                        try:
                            new_thr = float(new_thr_str)
                            if 0.0 <= new_thr <= 1.0:
                                per_animal[animal] = new_thr
                                print(f"[INFO] Threshold for {animal} updated to {new_thr:.3f}")
                            else:
                                print("[WARN] Threshold should be between 0 and 1.")
                        except ValueError:
                            print("[WARN] Invalid number.")
                else:
                    print("[WARN] Invalid index.")
            except ValueError:
                print("[WARN] Unknown option.")

    # Save thresholds
    data = {
        "default_threshold": default_threshold,
        "per_animal": per_animal,
    }
    with open(THRESHOLDS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved thresholds to {THRESHOLDS_JSON}.")
    print("=== Threshold Editor Finished ===")


# -------------------- MAIN LIVE DETECTION (DANGER PRIORITY) --------------------


def main_live():
    interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)

    # Determine number of classes from output shape
    output_shape = output_details["shape"]
    if len(output_shape) == 2:
        num_classes = int(output_shape[1])
    else:
        num_classes = int(np.prod(output_shape[1:]))

    index_to_class = get_index_to_class_mapping(num_classes)
    ensure_log_file(LOG_PATH)

    # Derive base animals from mapping
    base_animals = sorted({pretty_animal_name(name) for name in index_to_class.values()})

    # Load threshold config (ensures all animals are present)
    default_threshold, per_animal_thresholds = load_thresholds_config(
        THRESHOLDS_JSON, DEFAULT_CONFIDENCE_THRESHOLD, base_animals
    )

    # Precompute which indices are dangerous classes
    danger_indices = [
        idx for idx, name in index_to_class.items()
        if DANGER_KEYWORD in name.lower()
    ]

    print(f"[INFO] Dangerous class indices: {danger_indices}")
    if not danger_indices:
        print("[WARN] No class names contain 'dang'. Buzzer will NEVER trigger. Check your mapping/JSON.")

    # Start webcam (Windows default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam (index 0).")
        return

    print("[INFO] Press 'q' to quit.")

    last_beep_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame from webcam.")
                break

            # Preprocess and predict
            input_tensor = preprocess_frame(frame, input_details)
            interpreter.set_tensor(input_details["index"], input_tensor)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details["index"])[0]

            # --- 1) Always compute top-1 (for info) ---
            top_idx = int(np.argmax(preds))
            top_conf = float(preds[top_idx])
            top_raw_name = index_to_class.get(top_idx, f"class_{top_idx}")

            # --- 2) Search dangerous classes FIRST (priority) ---
            best_danger_idx = None
            best_danger_conf = 0.0
            best_danger_thr = None

            for d_idx in danger_indices:
                class_name = index_to_class[d_idx]
                animal_name_d = pretty_animal_name(class_name)
                thr_d = per_animal_thresholds.get(animal_name_d, default_threshold)
                conf_d = float(preds[d_idx])

                if conf_d >= thr_d and conf_d > best_danger_conf:
                    best_danger_idx = d_idx
                    best_danger_conf = conf_d
                    best_danger_thr = thr_d

            if best_danger_idx is not None:
                # Prioritize dangerous class even if human/top-1 is higher
                raw_class_name = index_to_class[best_danger_idx]
                confidence = best_danger_conf
                animal_name = pretty_animal_name(raw_class_name)
                used_threshold = best_danger_thr
                danger = True
            else:
                # No dangerous class above its threshold: fall back to top-1
                raw_class_name = top_raw_name
                confidence = top_conf
                animal_name = pretty_animal_name(raw_class_name)
                used_threshold = per_animal_thresholds.get(animal_name, default_threshold)
                danger = False

            # --- Overlay text ---
            label_text = f"{raw_class_name} ({confidence*100:.1f}%)"
            color = (0, 0, 255) if danger else (0, 255, 0)  # red for danger, green for safe

            cv2.putText(
                frame,
                label_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

            # Show threshold + top-1 info for debugging
            thr_text = f"Thr: {used_threshold*100:.0f}%"
            cv2.putText(
                frame,
                thr_text,
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            top_text = f"Top: {top_raw_name} ({top_conf*100:.1f}%)"
            cv2.putText(
                frame,
                top_text,
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            if danger:
                cv2.putText(
                    frame,
                    "DANGER!",
                    (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                now = time.time()
                if now - last_beep_time >= BEEP_MIN_INTERVAL:
                    print(
                        f"\n[DANGER] {raw_class_name} detected with "
                        f"{confidence*100:.2f}% (thr={used_threshold*100:.2f}%) "
                        f"[top was {top_raw_name} {top_conf*100:.2f}%]"
                    )
                    play_buzzer_async()
                    log_buzzer_event(LOG_PATH, animal_name, confidence, raw_class_name)
                    last_beep_time = now
            else:
                # Show live predictions for debugging
                print(
                    f"[SAFE] top={top_raw_name} {top_conf*100:.2f}%, "
                    f"chosen={raw_class_name} {confidence*100:.2f}%, thr={used_threshold*100:.2f}%",
                    end="\r",
                )

            # Show the frame
            cv2.imshow("AnimaRepel - Live View (TFLite)", frame)

            # Exit if user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Stopped. Webcam released.")


# -------------------- ENTRY POINT --------------------


def main():
    parser = argparse.ArgumentParser(description="AnimaRepel - Animal detector with per-animal thresholds")
    parser.add_argument(
        "--edit-thresholds",
        action="store_true",
        help="Open interactive threshold editor instead of running live detection",
    )
    args = parser.parse_args()

    if args.edit_thresholds:
        run_threshold_editor()
    else:
        main_live()


if __name__ == "__main__":
    main()
