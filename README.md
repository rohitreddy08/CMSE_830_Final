# CMSE_830_Midterm
Midterm project analyzing property interaction data

# Property Interaction Prediction Model

This project aims to predict property interactions based on historical data and analyze trends in user requests for properties. Various data processing, visualization, and regression modeling techniques are employed to explore the relationships between property features and user interactions.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Modeling Techniques](#modeling-techniques)
- [Visualizations](#visualizations)
- [Streamlit Application](#streamlit-application)


## Project Overview
This project utilizes property data, interaction logs, and associated metadata to build a predictive model that estimates the number of interactions a property will receive within a specific time frame (e.g., 3 days, 7 days).

The following steps are performed:
1. Data Cleaning and Preparation
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Modeling using Regression Techniques
5. Visualization of key insights
6. Deployment using a Streamlit app (optional)

## Dataset
The datasets used for this project include:
1. `property_data_set.csv`: Contains details about the properties, including age, size, rent, deposit, etc.
2. `property_interactions.csv`: Logs of interactions (clicks) on properties with timestamps.
3. `property_photos.tsv`: Metadata about property photos and URLs.

**Note**: The datasets need to be placed in the `data/` folder.

## Installation

1. Clone the repository:
```bash
   git clone https://github.com/yourusername/property-interactions-prediction.git
```

2. Navigate to the project directory:
```bash
   cd property-interactions-prediction
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```


4. Place the datasets in the data/ folder

## Project Structure

property-interactions-prediction/
│
├── data/
│   ├── property_data_set.csv
│   ├── property_interactions.csv
│   └── property_photos.tsv
│
├── notebooks/
│   ├── data_analysis.ipynb
│   └── model_evaluation.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── regression_models.py
│
├── images/
│   ├── property_trends.png
│   └── correlation_heatmap.png
│
├── README.md
├── requirements.txt
└── streamlit_app.py

## Usage

1. To explore the data, run the analysis notebooks:
    ``` jupyter notebook notebooks/data_analysis.ipynb ```

2. To run the model:
    ``` python src/regression_models.py ```

3. You can also explore the project interactively using the Streamlit app:
    ``` streamlit run streamlit_app.py ```

## Modeling Techniques

The project utilizes the following regression models to predict the number of property interactions within specific time frames (e.g., 3 days, 7 days):

1. **Linear Regression**:
   - This is a basic regression model that attempts to predict the target variable by fitting a linear relationship between the features and the number of interactions.
   - Linear regression works well for datasets where the relationship between features and the target variable is approximately linear.

2. **K-Nearest Neighbors (KNN) Regressor**:
   - KNN is a non-parametric model that predicts the target value based on the 'k' closest data points (neighbors) in the feature space.
   - It's useful when the data is not linear, and it adapts well to more complex relationships by leveraging local data patterns.

## Visualizations

Key visualizations include:

    Histograms showing the distribution of requests within the first 3 and 7 days.
    Correlation heatmaps for numeric features.
    Boxplots and scatter plots to identify outliers and trends.

## Streamlit Application

The project includes a Streamlit app for interactive exploration. [link].
