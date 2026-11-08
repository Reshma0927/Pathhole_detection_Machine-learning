# Pothole Detection Using ResNet-50

A beginner-friendly Python project that classifies a road image into two classes: `normal` and `pothole`, using transfer learning with ResNet-50. OpenCV is optional for later video processing; the included demo uses Streamlit and supports image upload or camera capture.

## Requirements

- Windows 10/11
- Python 3.10 (recommended for this requirements file)
- VS Code with the Python extension
- Internet connection for installing packages and downloading pretrained ResNet-50 weights
- A labeled pothole image dataset

## 1. Open the project

Extract the ZIP. In VS Code, select **File → Open Folder** and choose `Pothole-Detection-ResNet50`.

Open **Terminal → New Terminal**.

## 2. Create and activate a virtual environment (PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in the current terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If `py -3.10` is not found, install Python 3.10 from https://www.python.org/downloads/ and tick **Add Python to PATH** during setup. Reopen VS Code afterwards.

## 3. Install packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow installation fails, check that you are using 64-bit Python 3.10 and that the virtual environment is active.

## 4. Prepare your dataset

Put images in the following folders. The folder names must be exactly `normal` and `pothole`.

```text
dataset/
  train/
    normal/       # images of roads without potholes
    pothole/      # images containing potholes
  validation/
    normal/
    pothole/
```

Use different images in the training and validation folders. Do not put the same image in both. JPG, JPEG, and PNG images are suitable. Keep the dataset balanced where possible. This starter project does not include the dataset or a trained model.

## 5. Train the model

With `.venv` active, run:

```powershell
python train.py
```

Training can take a while and may require a lot of RAM. The first run downloads pretrained ResNet-50 weights. When training finishes, the model and class mapping are saved in `models/`.

## 6. Predict one image

Copy a road photo into `test_images`, for example `test_images/road.jpg`, then run:

```powershell
python predict.py test_images/road.jpg
```

Replace the filename with the actual image name.

## 7. Launch the visual demo

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`. Open that URL in your browser. Upload a road image or use the camera control, then click **Detect pothole**.

Stop the app with `Ctrl+C` in the terminal.

## Important notes

- The output is binary image classification. It does not draw bounding boxes or identify pothole coordinates.
- A real model requires labeled training images. Empty example folders are included only to show the structure.
- Accuracy depends on the dataset. Do not claim a particular accuracy until you train and evaluate the model on separate test images.
- The PDF supplied for this project mixes ResNet-50 classification with YOLOv3 object-detection instructions. This implementation follows the ResNet-50 title/objectives, not the YOLO/Darknet steps.
"# Pathhole_detection_Machine-learning" 
"# Pathhole_detection_Machine-learning" 
