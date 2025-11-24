import os
import time
import csv
import threading
from datetime import datetime
import platform
import subprocess
import json

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from picamera2 import Picamera2  # <-- Pi camera

# -------------------- CONFIG --------------------
MODEL_PATH = "object_identifier_model.tflite"   # TFLite model exported from Colab
BUZZER_SOUND = "buzzer.wav"                     # Path to buzzer sound file
LOG_PATH = "buzzer_log.csv"                     # CSV file to store buzzer events

# Optional: dataset root (only used if JSON is missing)
DATA_DIR = "AnimaRepel"

# JSON exported from Colab with train_generator.class_indices
CLASS_INDICES_JSON = "class_indices.json"

IMG_SIZE = 80                                   # Must match training
CONFIDENCE_THRESHOLD = 0.5                      # Detection threshold
DANGER_KEYWORD = "dang"                         # Any class name containing this is “dangerous”
BEEP_MIN_INTERVAL = 1.0                         # Seconds between buzzer beeps


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
    JSON should be of the form: { "WildBoar_dang": 6, "default_ndan": 8, ... }
    """
    if not os.path.exists(json_path):
        return None

    with open(json_path, "r") as f:
        class_indices = json.load(f)  # name -> index
    index_to_class = {}

    for name, idx in class_indices.items():
        index_to_class[int(idx)] = name

    # Ensure all indices 0..num_classes-1 are covered
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
      - ffplay (from ffmpeg) on Linux
    """
    if not os.path.exists(BUZZER_SOUND):
        print(f"[WARN] Buzzer sound file not found: {BUZZER_SOUND}")
        return

    def _play():
        try:
            system = platform.system()
            if system == "Windows":
                import winsound
                winsound.PlaySound(BUZZER_SOUND, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", BUZZER_SOUND],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as e:
            print(f"[ERROR] Could not play buzzer: {e}")

    t = threading.Thread(target=_play, daemon=True)
    t.start()


def is_dangerous_class(class_name: str, confidence: float) -> bool:
    """
    Decide if a predicted class is harmful enough to trigger the buzzer.
    We treat any class whose name contains 'dang' (case-insensitive)
    and has confidence above the threshold as dangerous.
    """
    return (DANGER_KEYWORD in class_name.lower()) and (confidence >= CONFIDENCE_THRESHOLD)


def pretty_animal_name(class_name: str) -> str:
    """
    Convert class name like 'WildBoar_dang' or 'WildBoar_ndan' into 'WildBoar'
    for nicer logging. If no underscore, return as-is.
    """
    parts = class_name.split("_")
    if len(parts) > 1:
        return "_".join(parts[:-1])
    return class_name


def preprocess_frame_rgb(frame_rgb, input_details):
    """
    frame_rgb: numpy array from Picamera2 (RGB888).
    Convert to model input: resized, normalized, correct dtype and shape.
    """
    resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    arr = resized.astype("float32") / 255.0

    # Ensure shape matches the model's expected input
    if len(input_details["shape"]) == 4:
        arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)

    # Cast to correct dtype
    arr = arr.astype(input_details["dtype"])
    return arr


# -------------------- MAIN LOOP (PICAMERA2) --------------------


def main():
    interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)

    # Determine number of classes from output shape
    output_shape = output_details["shape"]
    if len(output_shape) == 2:
        num_classes = int(output_shape[1])
    else:
        num_classes = int(np.prod(output_shape[1:]))

    index_to_class = get_index_to_class_mapping(num_classes)
    ensure_log_file(LOG_PATH)

    # Print which classes will trigger the buzzer
    danger_classes = [name for idx, name in index_to_class.items()
                      if DANGER_KEYWORD in name.lower()]
    print(f"[INFO] Classes that will trigger buzzer (contain '{DANGER_KEYWORD}'):", danger_classes)
    if not danger_classes:
        print("[WARN] No class names contain 'dang'. Buzzer will NEVER trigger. Check your mapping/JSON.")

    # ---- Picamera2 setup ----
    print("[INFO] Initializing Picamera2...")
    picam2 = Picamera2()

    # 640x480 RGB is enough for inference and display
    camera_config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(camera_config)
    picam2.start()
    print("[INFO] Picamera2 started. Press 'q' to quit.")

    last_beep_time = 0.0

    try:
        while True:
            # Capture frame as RGB array
            frame_rgb = picam2.capture_array()
            if frame_rgb is None:
                print("[WARN] Failed to capture frame from Picamera2.")
                continue

            # Preprocess for model
            input_tensor = preprocess_frame_rgb(frame_rgb, input_details)
            interpreter.set_tensor(input_details["index"], input_tensor)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details["index"])[0]

            idx = int(np.argmax(preds))
            confidence = float(preds[idx])
            raw_class_name = index_to_class.get(idx, f"class_{idx}")
            animal_name = pretty_animal_name(raw_class_name)

            danger = is_dangerous_class(raw_class_name, confidence)

            # Text to overlay
            label_text = f"{raw_class_name} ({confidence*100:.1f}%)"
            color = (0, 0, 255) if danger else (0, 255, 0)  # red for danger, green for safe

            # Work on a copy for display: convert RGB -> BGR for OpenCV window
            display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            cv2.putText(
                display_frame,
                label_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

            if danger:
                cv2.putText(
                    display_frame,
                    "DANGER!",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                now = time.time()
                if now - last_beep_time >= BEEP_MIN_INTERVAL:
                    print(f"\n[DANGER] {raw_class_name} detected with {confidence*100:.2f}%")
                    play_buzzer_async()
                    log_buzzer_event(LOG_PATH, animal_name, confidence, raw_class_name)
                    last_beep_time = now
            else:
                print(f"[SAFE] idx={idx}, class={raw_class_name}, conf={confidence*100:.2f}%", end="\r")

            # Show window (if you are running with GUI)
            cv2.imshow("AnimaRepel - Pi Camera (TFLite)", display_frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("\n[INFO] Stopped. Camera released.")


if __name__ == "__main__":
    main()