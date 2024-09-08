# Lonely Woman Detection System

This project is designed to detect a "lonely woman" from a camera feed using a pre-trained gender classification model and real-time people detection.

## Table of Contents
- Prerequisites
- Setup Instructions
- Running the System
- Project Structure

---

## Prerequisites

Before setting up the project, ensure you have the following installed on your system:

- **Python 3.x**
- **Virtual Environment Tool** (e.g., `venv` or `virtualenv`)
- **Git** (optional, for version control)

---

## Setup Instructions

### 1. Clone the Repository

If you haven't already cloned the repository, use the following command:

```
git clone <your-repository-url>
cd <repository-name>
```

### 2. Set Up Virtual Environment

Create and activate a virtual environment:

#### For Windows:
```
python -m venv venv
venv\Scripts\activate
```

#### For Linux/macOS:
```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

After activating the virtual environment, install the required Python packages by running:

```
pip install -r requirements.txt
```

Make sure that all the necessary libraries (e.g., `opencv-python`, `tensorflow`, etc.) are listed in the `requirements.txt`.

### 4. Add Pre-trained Model

Place the pre-trained gender classification model file (`gender_model.hdf5`) in the project root directory.

---

## Running the System

To run the detection system, use the following command:

```
python main.py
```

The system will start capturing the live feed from your webcam, and it will detect whether there is a lonely woman in the frame. If detected, a message will be displayed on the video feed.

---

## Project Structure

```
|-- main.py                 # Main Python file that runs the detection system
|-- venv/                   # Virtual environment directory
|-- gender_model.hdf5        # Pre-trained gender classification model
|-- requirements.txt         # List of required Python dependencies
|-- README.md                # Project documentation (this file)
```

---

## License

This project is licensed under the MIT License.
