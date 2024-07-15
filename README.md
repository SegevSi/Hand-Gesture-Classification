
### README.md

```markdown
# Hand Gesture Classification

This project implements a real-time hand gesture classification system using Python, MediaPipe, and TensorFlow. It captures video input from a webcam, processes the hand landmarks, and predicts gestures based on a trained model.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Adding More Classes](#adding-more-classes)
- [License](#license)

## Features
- Real-time hand gesture classification
- Supports multiple hand gestures
- Customizable gesture classes
- Uses MediaPipe for hand landmark detection
- Trained with TensorFlow for efficient predictions

## Requirements
To run this project, you'll need the following Python packages:
- `mediapipe`
- `opencv-python`
- `tensorflow`
- `pandas`
- `scikit-learn`
- `numpy`

You can install these packages using pip:
```bash
pip install mediapipe opencv-python tensorflow pandas scikit-learn numpy
```

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/SegevSi/hand-pose-classification.git
   cd hand-pose-classification
   ```

2.  Start the interface for real-time prediction:
   ```bash
   python interface.py
   ```

## Usage
After setting up, you can start the webcam and perform gestures in front of the camera. The predicted gesture will be displayed on the screen.




### Adding More Classes

To add more classes of poses:

1. **Update the dataset collection:**
   - In `create_hand_poses_datasets.py`, call `create_hand_pose_dataset` with a new class name and class number. For example:
     ```python
     create_hand_pose_dataset("new_gesture_data", 5, num_samples)
     ```

2. **Modify the model training:**
   - In `create_clc_model.ipynb`, include the new CSV file in the `files` list:
     ```python
     files = ["dislikes_data.csv", "likes_data.csv", "middle_finger_data.csv", "gun_data.csv", "all_good_data.csv", "new_gesture_data.csv"]
     ```

3. **Save the trained model with a different name:**
   - After training, save the model with a new filename to avoid overwriting the existing model:
     ```python
     model.save("hand_pose_clc_v2.keras")
     ```

4. **Update the model path in the interface:**
   - In `interface.py`, change the model loading line to point to the new model file:
     ```python
     pose_clc = tf.keras.models.load_model("hand_pose_clc_v2.keras")
     ```

5. **Train the model again to include the new classes** by running the notebook.

6. **Update the `solutions_dict` in `interface.py`** to map the new class index to its gesture name:
   ```python
   solutions_dict = {0: "dislike", 1: "like", 2: "middle finger", 3: "gun", 4: "ok", 5: "new gesture"}
   ```

## License
This project is licensed under the MIT License. See the LICENSE file for details.
```
