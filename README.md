# ASL Sign Language Detection

Real-time American Sign Language (ASL) detection, gesture classification, and alphabet recognition using OpenCV, Computer Vision, and Scikit-Learn.

## Quick Start (Run Immediately)

To launch the sign language detection webcam application directly:

```powershell
python "Sign language detection.py"
```
or
```powershell
python app.py
```

---

## Features

- **Live Sign Recognition & Geometric Hand Tracking**:
  - Detects and tracks hand contour, palm center (via distance transform), fingertips, and finger webbing defects.
  - Recognizes ASL alphabet letters and gestures:
    - **[A]**: Fist (thumb at side)
    - **[B]**: 4 fingers straight up (thumb in)
    - **[C]**: Curved C-shape
    - **[D]**: Index finger pointing up
    - **[F]**: OK sign (index + thumb loop, 3 fingers up)
    - **[I]**: Pinky finger up only
    - **[L]**: Index up and Thumb out at 90° angle
    - **[U]**: Index and Middle fingers touching together
    - **[V]**: Peace / V-shape (Index and Middle spread)
    - **[W]**: 3 fingers up (Index, Middle, Ring)
    - **[Y]**: Thumb and Pinky spread wide
    - **[5]**: Open Palm (all 5 fingers spread)
- **Interactive Instruction Panel (Sidebar)**:
  - Displays a live **Sign Reference Guide** showing how to perform each sign.
  - Features **live active highlighting**: when you perform a sign, the matching row in the instruction panel highlights in vibrant green with `>> [V] Peace sign`!
- **Sentence Builder / Translator**:
  - Holding a sign steadily for ~0.8 seconds types its letter into the message.
  - Spell words such as `HI`, `BYE`, or `LIKE`; matching common-word suggestions appear as you sign.
  - Press `[Tab]` to complete the current word with the first suggestion, `[Space]` to start the next word, and `[Backspace]` to edit.
  - This spells words from static alphabet signs; it does not recognize whole-word or moving ASL signs.
- **HUD & Silhouette Thumbnail**:
  - Displays a high-tech HUD overlay with detected sign, gesture description, palm center, fingertips, and a $28 \times 28$ normalized silhouette inset thumbnail.

---

## Interactive Controls

| Key | Action |
| :--- | :--- |
| **Q** or **Esc** | Quit application |
| **S** | Save screenshot (`screenshot_<timestamp>.png`) |
| **P** | Save the detected hand crop and print the sign/result |
| **H** | Toggle Instruction Panel on/off |
| **M** | Toggle binary hand segmentation mask view |
| **Tab** | Complete current word from common suggestions |
| **Space** | Start the next word |
| **Backspace** | Delete last typed letter |
| **R** or **C** | Reset / Clear translated text |

For more reliable hand capture, keep one hand centered inside the box against a
plain background with good lighting. Use **M** to inspect the segmentation mask
if the hand is not being detected cleanly.

---

## Command-Line Subcommands

The application also supports optional CLI subcommands:

1. **Run Webcam Recognition:**
   ```powershell
   python app.py webcam
   ```

2. **Train Model on Dataset:**
   ```powershell
   python app.py train
   ```

3. **Predict Single 28x28 Image:**
   ```powershell
   python app.py predict-image path/to/image.png
   ```

4. **Run Unit Tests:**
   ```powershell
   python -m unittest -v
   ```

---

## Project Structure

- `Sign language detection.py`: Compatibility entry point to launch the live sign detection application.
- `app.py`: Core application code containing computer vision pipeline, hand geometry analysis, instruction panel renderer, model training, and webcam inference.
- `models/`: Stores the trained classifier model (`sign_language_model.joblib`).
- `datasets/`: Contains dataset CSV files (`sign_mnist_train.csv`, `sign_mnist_test.csv`).
- `test_app.py`: Unit test suite (13 tests) verifying data loading, training, model persistence, image preprocessing, and instruction panel rendering.
- `requirements.txt`: Python package requirements.
