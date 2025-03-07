# Ratings Prediction using Neural Networks and Logistic Regression

This repository contains a Python script that demonstrates how to predict ratings using both a Feedforward Neural Network (NN) and Logistic Regression (Logit). The dataset used is stored in an Excel file (`Ratings_exercise.xlsx`), and the script performs a 5-fold cross-validation to evaluate the performance of both models.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [License](#license)

## Installation

To run this script, you need to have Python installed along with the following libraries:

- `polars`
- `pandas`
- `numpy`
- `torch`
- `scikit-learn`
- `matplotlib`

You can install these libraries using pip:

```bash
pip install polars pandas numpy torch scikit-learn matplotlib

## Usage

Clone the repository:
bash
Copy
git clone https://github.com/yourusername/ratings-prediction.git
cd ratings-prediction
Place your Ratings_exercise.xlsx file in the root directory of the project.
Run the Jupyter Notebook or Python script:
bash
Copy
jupyter notebook ratings_prediction.ipynb
or

bash
Copy
python ratings_prediction.py
Methodology

## Data Preparation

The dataset is loaded using polars and then converted to a pandas DataFrame.
Features (rel_size, excess_rets, idio_stdev, ni_ta, tl_ta) and target (ratings2) are extracted.
The data is split into training and testing sets using an 80-20 split.
Model Training

Feedforward Neural Network (NN):
A simple feedforward neural network with one hidden layer is implemented using PyTorch.
The model is trained using the Binary Cross-Entropy Loss with Logits and optimized using the Adam optimizer.
Logistic Regression (Logit):
A logistic regression model is trained using scikit-learn.
Evaluation

The performance of both models is evaluated using 5-fold cross-validation.
The Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC) curve is calculated for each fold.
The average AUC for both models is reported.
Results

The script generates ROC curves for each fold, comparing the performance of the Neural Network and Logistic Regression models. The average AUC for both models is printed at the end of the script.

## Example output:


Average Neural Network AUC: 0.XXX
Average Logistic Regression AUC: 0.XXX
License

This project is licensed under the GNU License - see the LICENSE file for details.

