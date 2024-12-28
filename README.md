# Simplistic Quantum Machine Learning (QML) Implementation on Synthetic Data

**Author:** Leo Martinez III - [LinkedIn](https://www.linkedin.com/in/leo-martinez-iii/)

**Contact:** [leo.martinez@students.tamuk.edu](mailto:leo.martinez@students.tamuk.edu)

**Created:** Fall 2024

To clone this repository:  
```
git clone https://github.com/LeoMartinezTAMUK/Quantum_Machine_Learning.git
```

---

## Overview  

This project implements a **Quantum Machine Learning (QML)** model using **Pennylane** to classify synthetic data. It leverages quantum variational circuits with angle encoding and gradient-based optimization to perform binary classification tasks. Dimensionality reduction and feature scaling techniques are used to prepare the data for quantum processing.  

---

## Features  

- **Dimensionality Reduction:** Utilizes Principal Component Analysis (PCA) to reduce input features to fit quantum processing constraints.  
- **Quantum Circuit Design:** Implements angle encoding and strongly entangling layers for quantum feature processing.  
- **Optimization:** Employs the Adam optimizer to train a hybrid quantum-classical variational classifier.  
- **Metrics:** Outputs evaluation metrics such as precision, recall, F1-score, and balanced accuracy.  

---

## Installation  

### Requirements:  

- **Python**: 3.12.6  
- **Libraries**:
  - `pennylane`
  - `numpy`
  - `scikit-learn`
  - `pandas`  

---

## Workflow  

### 1. Data Preparation  
The project starts by reading a synthetic dataset (`synthetic_data.csv`). PCA reduces the dataset dimensions to 2, and the data is standardized to fit the quantum processing requirements.  

### 2. Quantum Circuit Design  
The quantum circuit uses:  
- **Angle Encoding:** Maps classical data to quantum states.  
- **Strongly Entangling Layers:** Introduces correlations between qubits.  

### 3. Training and Optimization  
- A variational quantum classifier predicts labels by combining quantum circuits with a classical bias term.  
- The classifier minimizes the square loss function using the **Adam Optimizer** with a batch size of 10.  

### 4. Evaluation  
The trained model evaluates the test dataset, outputting metrics such as:  
- Precision  
- Recall  
- F1 Score  
- Balanced Accuracy  

---

## Usage  

1. **Prepare the Dataset:**  
   Place your synthetic dataset as `data/synthetic_data.csv`.  

2. **Run the Script:**  

3. **View Results:**  
The script will print training progress, final test accuracy, and classification metrics.  

---

## Example Output  


Here’s the raw text for the README file:

makefile
Copy code
# Basic Quantum Machine Learning (QML) Implementation on Synthetic Data

**Author:** Leo Martinez III  
**Created:** Fall 2024  

To clone this repository:  
git clone https://github.com/LeoMartinezTAMUK/QML_Project.git

yaml
Copy code

---

## Overview  

This project implements a **Quantum Machine Learning (QML)** model using **Pennylane** to classify synthetic data. It leverages quantum variational circuits with angle encoding and gradient-based optimization to perform binary classification tasks. Dimensionality reduction and feature scaling techniques are used to prepare the data for quantum processing.  

---

## Features  

- **Dimensionality Reduction:** Utilizes Principal Component Analysis (PCA) to reduce input features to fit quantum processing constraints.  
- **Quantum Circuit Design:** Implements angle encoding and strongly entangling layers for quantum feature processing.  
- **Optimization:** Employs the Adam optimizer to train a hybrid quantum-classical variational classifier.  
- **Metrics:** Outputs evaluation metrics such as precision, recall, F1-score, and balanced accuracy.  

---

## Installation  

### Requirements:  

- **Python**: 3.12.6  
- **Libraries**:
  - `pennylane`
  - `numpy`
  - `scikit-learn`
  - `pandas`  

Install dependencies using:  
pip install -r requirements.txt

yaml
Copy code

---

## Workflow  

### 1. Data Preparation  
The project starts by reading a synthetic dataset (`synthetic_data.csv`). PCA reduces the dataset dimensions to 2, and the data is standardized to fit the quantum processing requirements.  

### 2. Quantum Circuit Design  
The quantum circuit uses:  
- **Angle Encoding:** Maps classical data to quantum states.  
- **Strongly Entangling Layers:** Introduces correlations between qubits.  

### 3. Training and Optimization  
- A variational quantum classifier predicts labels by combining quantum circuits with a classical bias term.  
- The classifier minimizes the square loss function using the **Adam Optimizer** with a batch size of 10.  

### 4. Evaluation  
The trained model evaluates the test dataset, outputting metrics such as:  
- Precision  
- Recall  
- F1 Score  
- Balanced Accuracy  

---

## Usage  

1. **Prepare the Dataset:**  
   Place your synthetic dataset as `data/synthetic_data.csv`.  

2. **Run the Script:**  
python qml_classification.py

yaml
Copy code

3. **View Results:**  
The script will print training progress, final test accuracy, and classification metrics.  

---

## Contributions  

Feel free to fork this repository, report issues, or contribute with enhancements.  

