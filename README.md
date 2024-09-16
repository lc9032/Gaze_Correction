# Real-Time Gaze Correction in Video Conferencing Using Deep Learning

This project implements a Gaze Redirection system.
The system processes datasets for training, trains a gaze redirection model, and provides real-time gaze correction using a webcam or video feed.


## Requirements

### Environmental setup:

Make sure to have the following CUDA, cuDNN, Python versions installed:

| Library      | Version  |
|--------------|----------|
| CUDA         | 11.8     |
| cuDNN        | 8.6.0    |
| Python       | 3.11.9   |

### Required packages:

Ensure you have the necessary dependencies installed. You can install the required libraries using requirements.txt or manually install them.

| Library      | Version  |
|--------------|----------|
| TensorFlow   | 2.13.1   |
| OpenCV       | 4.10.0   |
| Mediapipe    | 0.10.14  |
| PyVirtualCam | 0.11.1   |
| NumPy        | 1.24.3   |

```python
pip install -r requirements.txt
```

---

## Usage

### Dataset Processing
The dataset processing is handled by `processingDataset.py`. This script is responsible for loading and preprocessing various gaze datasets that are used to train the model.

#### Supported Datasets

The script supports different datasets via a `DATASET` switch, which needs to be set based on the dataset being used.

- `DATASET = 0`: **Columbia Gaze Data Set**
- `DATASET = 1`: **DIRL Gaze Dataset**
- `DATASET = 2`: **U2Eyes Database**

You need to modify the `DATASET` variable in `processingDataset.py` to use the corresponding dataset.

```python
DATASET = 2  # Change this value to 0, 1, or 2 depending on the dataset
```


Once you have set the appropriate dataset, the script will:

1. Load images and corresponding landmarks, gaze information, etc.
2. Normalize and resize images.
3. Create input pairs for training the gaze redirection model.

---

### Training the Model
The model training is managed via train.py. It uses the preprocessed datasets and trains the model to redirect gaze in the images.

#### Steps for Training
1. Dataset Setup: Make sure you’ve preprocessed the correct dataset as explained in the Dataset Processing section.
2. Model Training: Run the train.py script to start training.

```python
python train.py
```

3. Checkpointing: The script automatically saves model checkpoints at each epoch for easy resumption of training.

#### Model Parameters
You can adjust key training parameters such as batch_size, learning_rate, and the number of epochs in train.py:

```python
self.batch_size = 256  # Adjust the batch size
self.epochs = 200      # Set the number of epochs
```

---

### Real-Time Gaze Correction System

The real-time gaze correction system is implemented in gaze_corr_sys.py. It uses the trained model to correct the gaze of a person in a webcam feed or video stream in real time.

#### Key Features
- Webcam Integration: The system captures frames from the webcam, applies the trained model to adjust the gaze direction, and displays the modified frames.
- Virtual Camera Support: You can use the pyvirtualcam library to output the adjusted video stream to a virtual camera, which can be used in video calls or other applications.

### Running the Real-Time System
Make sure the model has been trained and saved as a checkpoint. Then, you can run the real-time gaze correction system as follows:

```python
python gaze_corr_sys.py
```
