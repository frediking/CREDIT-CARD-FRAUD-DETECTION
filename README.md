# Credit Card Fraud Detection

![Screenshot 2025-04-04 at 14 43 30](https://github.com/user-attachments/assets/f80e600b-07ad-4896-af30-07f3d7e8e380)

*NB: the comma separated numeric values or features seen and used in the screenshot above are from the dataset used for this project.*
*particularly at index 26 of the dataset.*

*LINK TO THE DATASET IS PROVIDED AT THE END OF THIS README.*



This project uses machine learning to detect potential fraudulent credit card transactions. It includes a Flask web application for interactive prediction, a trained RandomForestClassifier model, and Jupyter Notebooks for exploratory data analysis, model training, and evaluation.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Author](#author)
- [Data Source](#data-source)

## Overview

The primary goal of this project is to build and deploy a model that can distinguish fraudulent transactions from legitimate ones. The project includes:
- **Data Preprocessing & Analysis:** Using Jupyter Notebooks to clean and explore the dataset.
- **Model Training:** A RandomForestClassifier is trained (with the help of techniques like SMOTE to address class imbalance) and evaluated using various classification metrics.
- **Web Application:** A Flask-based web app allows users to input transaction features and receive a prediction along with confidence scores.

## Project Structure

```
CC FRAUD DETECTION/
├── application.py         # Flask application serving model predictions
├── rfcc_model2.pkl        # Pre-trained RandomForestClassifier model
├── scalerrf.pkl           # Scaler used during training for model input
├── notebooks/
│   └── cc2.ipynb          # Jupyter Notebook with data analysis, model training, and evaluation
├── templates/
│   └── index.html         # HTML template for the interactive web interface
└── requirements.txt       # Python dependencies list
```

- **application.py:** The Flask server which serves the prediction endpoint.
- **templates/index.html:** The front-end form where users can input feature values.
- **notebooks/cc2.ipynb:** Contains the data exploration, feature engineering, model training, and validation code.
- **rfcc_model2.pkl & scalerrf.pkl:** Artifacts saved after training to be used for prediction.
- **requirements.txt:** Lists all required Python packages.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/cc-fraud-detection.git
   cd cc-fraud-detection
   ```

2. **Create and activate a virtual environment using Python 3:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *If you do not have a `requirements.txt` file yet, you can install the needed packages manually:*
   ```bash
   pip install flask numpy pandas scikit-learn joblib imbalanced-learn matplotlib seaborn jupyter
   ```

## Usage

### Running the Flask Application

1. **Start the server:**

   ```bash
   python3 application.py
   ```

   The server will start on a specified port (e.g., [http://127.0.0.1:8000](http://127.0.0.1:8000)). You can adjust the port in your `application.py` if needed.

2. **Access the Web Interface:**

   Open your browser and go to [http://127.0.0.1:8000](http://127.0.0.1:8000). Input the transaction features (as comma-separated numeric values) in the form and click the "Analyze Transaction" button to receive a prediction and associated confidence.

### Running the Jupyter Notebook

1. **Start Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

2. **Open the Notebook:**

   In your browser, navigate to the `notebooks/cc2.ipynb` file for an in-depth analysis and training process.

## Notebooks

- **cc2.ipynb:** This notebook contains code to:
  - Import and preprocess the dataset.
  - Explore the data visually with plots (using matplotlib and seaborn).
  - Address class imbalance using SMOTE.
  - Train a RandomForestClassifier with cross-validation.
  - Evaluate the model using metrics such as accuracy, precision, recall, F1-score, ROC curves, and more.

## Dependencies

The project primarily uses the following Python libraries:
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn
- Jupyter Notebook

*All dependencies are listed in the `requirements.txt` file.*

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to the contributors of scikit-learn, Flask, and the open source community for their valuable libraries and tools.
- Special thanks to [Any mentors or institutions] for support and guidance.

## Author
Fredinard Ohene-Addo

## Data Source
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

*all models are can be accessed at v1.0.0 releases*
