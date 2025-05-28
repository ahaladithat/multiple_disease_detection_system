# Multiple Disease Detection System Using Machine Learning

## Overview
This project implements a Multiple Disease Detection System using machine learning. It uses several pre-trained models to predict different diseases based on input data. The models are stored as `.pkl` files, and the project includes a main application (`app.py`) to run predictions.

## Project Structure
multiple_disease_detection/
├── app.py # Main application
├── disease_modules/ UI pages for individual disease detection(.py files)
├── models/ Trained machine learning models (.pkl files)
└── README.md # Project documentation

## Features
- Supports multiple diseases prediction using individual machine learning models.
- Easy to extend by adding new models or features.
- Modular code structure for maintainability.

## Getting Started

### Prerequisites
- Python 3.6 or higher

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multiple-disease-detection.git
2. Navigate to the project directory:
    cd multiple-disease-detection
3. (Optional) Create and activate a virtual environment:
    python -m venv venv source venv/bin/activate       # On Windows: venv\Scripts\activate
4. Install dependencies:
    pip install -r requirements.txt

### Run the application
    streamlit app.py
