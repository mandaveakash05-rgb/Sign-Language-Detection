# ASL Alphabet Sign Language Detection

This project trains a letter classifier from the **Sign Language MNIST** dataset,
then predicts either a dataset-style image or a hand held in a webcam's on-screen
box. It recognizes the 24 static letters in the dataset; **J** and **Z** are not
included because they require motion.

## 1. Install dependencies

Use Python 3.10 or newer and install:

```powershell
python -m pip install -r requirements-project.txt
```

`requirements-project.txt` is used because the existing OneDrive `requirements.txt`
could not be read while its cloud file provider was unavailable.

## 2. Add the dataset

Download the Sign Language MNIST CSV files from the
[Kaggle dataset page](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
or another trusted dataset host. The training file must be named
`sign_mnist_train.csv`; the optional test file must be named
`sign_mnist_test.csv`. Put them in a `datasets` folder:

```text
datasets/
  sign_mnist_train.csv
  sign_mnist_test.csv
```

The CSV format has a `label` column and 784 pixel columns for each 28x28 image.
Dataset files are not bundled with this project and must be obtained separately.

## 3. Train

```powershell
python app.py train
```

Training validates the input data, reports test accuracy when the optional test
CSV is present, and writes `models/sign_language_model.joblib`. To use different
paths:

```powershell
python app.py train --data-dir path\to\csv-files --model path\to\model.joblib
```

## 4. Predict

For an image already prepared as a 28x28 grayscale dataset-style image:

```powershell
python app.py predict-image path\to\image.png
```

To use a webcam:

```powershell
python app.py webcam
```

Put one clearly visible hand inside the on-screen box and sign letters steadily
to spell a message. Common suggestions include `HI`, `BYE`, and `LIKE`; press
**Tab** to complete the current word, **Space** to start the next word, and
**Backspace** to edit. This is alphabet spelling, not whole-word gesture
recognition. Keep the hand centered against a plain background with good
lighting; press **M** to inspect the segmentation mask if capture is poor.
Press **Q** or **Esc** to quit. Choose another camera with `--camera 1`.
Webcam predictions are approximate: the classifier is trained on dataset
images, and lighting, backgrounds, camera position, and signing style can
affect results.

## Run tests

```powershell
python -m unittest -v
```
