# Linear Regression for Graduate Admission Prediction

## Project Overview

This project implements a linear regression model to predict the "Chance of Admit" for graduate school applicants. The model is built primarily using fundamental linear algebra concepts and the Normal Equation for parameter estimation. The "Admission_Predict.csv" dataset is used for training and evaluation.

This project was undertaken as part of a Linear Algebra course to demonstrate the practical application of linear algebra theories in solving real-world-like problems.

## Core Concepts Demonstrated

* **Linear Algebra Fundamentals:**
    * Representation of data using vectors and matrices.
    * Matrix operations: transpose, multiplication, and inverse.
    * Solving systems of linear equations via the Normal Equation: $w = (X^T X)^{-1} X^T y$.
* **Mathematical Modeling:**
    * Formulating a linear regression model: $\hat{y} = Xw$.
    * Defining and using the Mean Squared Error (MSE) loss function.
* **Data Handling and Preparation:**
    * Loading and inspecting a CSV dataset.
    * Feature selection and preparation (including adding an intercept term).
    * Splitting data into training and testing sets.
* **Model Evaluation:**
    * Making predictions on unseen data.
    * Calculating and interpreting common regression metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared ($R^2$).

## Dataset

The dataset used is "Admission_Predict.csv", which contains features such as:
* GRE Scores
* TOEFL Scores
* University Rating
* SOP (Statement of Purpose) Strength
* LOR (Letter of Recommendation) Strength
* CGPA (Cumulative Grade Point Average)
* Research Experience

The target variable is "Chance of Admit ".

## Implementation Details

The core of the model parameter estimation relies on the Normal Equation, implemented using Python with libraries such as:
* **Pandas:** For data loading and manipulation.
* **NumPy:** For numerical computations, especially matrix operations.
* **Statsmodels (or similar logic):** For conveniently adding the constant/intercept term to the feature matrix.
* **Scikit-learn:** For splitting the data into training and testing sets and optionally for metrics calculation.

## How to Use

1.  **Ensure necessary libraries are installed:**
    ```bash
    pip install pandas numpy scikit-learn statsmodels
    ```
2.  **Place the `Admission_Predict.csv` file in the same directory as the script/notebook.**
3.  **Run the Python script or Jupyter Notebook.** The script will:
    * Load and preprocess the data.
    * Split the data into training and testing sets (90% train, 10% test).
    * Calculate the optimal weight vector `w` using the Normal Equation on the training data.
    * Make predictions on the test data.
    * Evaluate and print the model's performance metrics (MSE, RMSE, MAE, R-squared).

## Results

The script outputs the calculated weight vector `w` and the performance metrics on both the training and test sets. For this particular run, the test set performance was:
* Mean Squared Error (MSE): 0.0066
* Root Mean Squared Error (RMSE): 0.0810 
* Mean Absolute Error (MAE): 0.0571
* R-squared ($R^2$): 0.7691

These results indicate that approximately 76.91% of the variance in the 'Chance of Admit' can be explained by the selected features using this linear model.

## Future Work (Optional)

* Explore the impact of different feature selections.
* Implement regularization techniques (e.g., Ridge Regression) if multicollinearity is suspected or to improve generalization.
* Compare the Normal Equation approach with iterative methods like Gradient Descent.
