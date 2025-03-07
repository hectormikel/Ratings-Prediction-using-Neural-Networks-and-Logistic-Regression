# Ratings Prediction using Neural Networks and Logistic Regression

This repository contains a Jupyter Notebook that implements a machine learning model to predict ratings based on various financial metrics. The model uses both a FeedForward Neural Network and Logistic Regression for comparison.

## Overview

The notebook performs the following tasks:

1. **Data Loading**: Reads data from an Excel file named `Ratings_exercise.xlsx`.
2. **Data Preprocessing**:
   - Selects relevant features and target variable.
   - Splits the data into training and testing sets.
   - Scales the features using `StandardScaler`.
3. **Model Training**:
   - Implements a FeedForward Neural Network using PyTorch.
   - Uses K-Fold Cross-Validation to train and evaluate the model.
   - Compares the performance of the Neural Network with Logistic Regression.
4. **Evaluation**:
   - Calculates the Area Under the Curve (AUC) for both models.
   - Plots the Receiver Operating Characteristic (ROC) curves for visual comparison.

## Dependencies

- `polars`
- `pandas`
- `numpy`
- `torch`
- `scikit-learn`
- `matplotlib`

## Usage

1. Ensure you have the required libraries installed. You can install them using:
   ```bash
   pip install polars pandas numpy torch scikit-learn matplotlib
   ```

2. Place the `Ratings_exercise.xlsx` file in the same directory as the notebook.

3. Run the notebook cells sequentially to train the models and visualize the results.

## Results

The notebook outputs the average AUC for both the Neural Network and Logistic Regression models across all folds. It also displays ROC curves for each fold, allowing for a visual comparison of model performance.

## Contributing

Feel free to fork this repository and submit pull requests for any improvements or additional features.

## License

This project is licensed under the GNU License.
