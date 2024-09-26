# Employee Attrition Prediction Using Decision Tree and Random Forest Models

This project aims to predict employee attrition using classification models. Two machine learning models, **Decision Tree** and **Random Forest**, are used to develop a tool that can help HR professionals anticipate employee departures and take proactive measures to improve retention. An interactive R Shiny application is designed to allow users to input data, tune model parameters, and predict attrition.

## Project Overview

- **Objective**: Predict employee attrition based on features such as age, monthly income, overtime, job satisfaction, and more.
- **Data Source**: The dataset contains 487 observations and 34 features, including both continuous and categorical variables. The target variable is `Attrition`, which indicates whether an employee will leave the organization (1) or not (0).
- **Tech Stack**: R, R Shiny, Random Forest, Decision Tree, Data Visualization (Histograms, Bar Charts)

## Dataset

The dataset includes various employee attributes such as age, gender, working years, salary hike, and job satisfaction. The target variable is binary (`Attrition`), indicating whether an employee is likely to leave the company.

### Dataset Files:
- **`employee_attrition.csv`**: Contains 487 rows and 34 features for model training and testing.
- **`unseen_data.csv`**: New data that users can upload via the app for making predictions using the pre-trained models.

## File Structure

- **`Code.R`**: The R script containing code for data pre-processing, model training, and implementation in the R Shiny application.
- **`Report.pdf`**: Detailed report discussing the methodology, model performance, and user guide for the R Shiny app.
- **`employee_attrition.csv`**: Dataset used for training and testing.
- **`unseen_data.csv`**: Dataset used for testing the app's prediction feature.
- **`requirements.txt`**: R package dependencies required to run the app.

## Model and Performance

Two classifiers were employed to predict employee attrition:
1. **Random Forest (RF)**:
   - Accuracy: 64%
   - Precision: 60%
   - Recall: 70%
   - F1 Score: 0.70
   - Key features: `MonthlyIncome`, `Overtime`, `Age`

2. **Decision Tree (DT)**:
   - Accuracy: 67%
   - Precision: 72%
   - Recall: 69%
   - F1 Score: 0.65
   - Key features: `MonthlyIncome`, `Overtime`, `JobSatisfaction`

Both models provide detailed evaluation through confusion matrices, precision, recall, F1 scores, and accuracy metrics. Random Forest offers higher overall accuracy, while the Decision Tree offers greater interpretability.

## Usage

The project provides a user-friendly **R Shiny** interface that allows users to:
1. Upload a CSV file containing employee data.
2. Choose between two machine learning models: Decision Tree or Random Forest.
3. Tune hyperparameters for both models (e.g., `mtry`, `number of trees` for Random Forest, and `min samples` for Decision Tree).
4. Predict employee attrition and export the results for further analysis.

## Installation

To run the project locally:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Employee-Attrition-Prediction.git
    cd Employee-Attrition-Prediction
    ```

2. Install the required R packages:
    ```r
    install.packages(c("shiny", "randomForest", "rpart", "ggplot2"))
    ```

3. Run the R Shiny application:
    ```r
    shiny::runApp("Code.R")
    ```

## Acknowledgements

This project was developed as part of the MSc in Business Analytics program at Bayes Business School.

## License

MIT License
