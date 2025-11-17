# AnimaRepel – Harmful Animal Detector

Uses a trained TFLite model + webcam.
When a “dangerous” class (name contains `dang`) is detected:

* Plays `buzzer.wav`
* Appends a row to `buzzer_log.csv` with: time, animal, confidence, raw class name

---

## 1. Folder Contents

Put these files in **one folder**, for example `AnimaRepel`:

* `AnimaRepel.py`
* `object_identifier_model.tflite`
* `class_indices.json`
* `buzzer.wav`
* *(optional)* `AnimaRepel/` (dataset folder with class subfolders)

---

## 2. Windows – Step by Step

### 2.1 Install Python

1. Download Python 3 from: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Run installer:

   * Tick **“Add Python to PATH”**
   * Click **Install Now**

Check:

```bat
python --version
```

(or)

```bat
py --version
```

---

### 2.2 Create virtual environment

**Open Command Prompt in the project folder:**

* In File Explorer → go to your `AnimaRepel` folder
* Click address bar → type `cmd` → Enter

**Run:**

```bat
py -3 -m venv .venv
.\.venv\Scripts\activate
```

Prompt should start with `(.venv)`.

---

### 2.3 Install Python packages

```bat
python -m pip install --upgrade pip
python -m pip install tensorflow opencv-python numpy
```

---

### 2.4 Run AnimaRepel

```bat
python AnimaRepel.py
```

* Webcam window opens.
* When a class whose name contains `dang` is detected with enough confidence:

  * “DANGER!” text appears
  * `buzzer.wav` plays
  * `buzzer_log.csv` is updated

Press **`q`** in the video window to quit.

---

## 3. Linux (Ubuntu/Debian) – Step by Step

> Use a **normal Linux desktop** (not WSL) so webcam + GUI work.

### 3.1 Install system packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg
```

Check:

```bash
python3 --version
```

---

### 3.2 Create virtual environment

In a terminal, inside the `AnimaRepel` folder:

```bash
cd /path/to/AnimaRepel

python3 -m venv .venv
source .venv/bin/activate
```

Prompt should start with `(.venv)`.

---

### 3.3 Install Python packages

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow opencv-python numpy
```

---

### 3.4 Run AnimaRepel

```bash
python AnimaRepel.py
```

* Webcam window opens.
* On detecting a `*dang` class with enough confidence:

  * “DANGER!” on screen
  * `buzzer.wav` plays (via `ffplay` from `ffmpeg`)
  * `buzzer_log.csv` updated in the same folder

Press **`q`** in the video window to quit.

---

## 4. Notes

* Dangerous classes are those whose name contains `dang` (case-insensitive), e.g. `WildBoar_dang`.
* `class_indices.json` must match the training (`train_generator.class_indices` from Colab).
* If no class names contain `dang`, the buzzer will never trigger.
