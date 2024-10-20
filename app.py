import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import io

# Set up the page configuration
st.set_page_config(page_title="Property Data Analysis", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", [
    "Introduction",
    "Data Upload",
    "Data Information",
    "Data Visualization",
    "Model Training and Evaluation"
])

# Add GitHub link below navigation
st.sidebar.markdown("""
---
### 📂 **Download Data Files**
[GitHub Repository](https://github.com/rohitreddy08/CMSE_830_Midterm/tree/main/Data)
""")

# Function to correct corrupted JSON and get count of photos
def correction(x):
    if pd.isnull(x) or x == 'NaN':
        return 0
    else:
        try:
            corrected_x = x.replace('\\', '').replace('{title', '{"title').replace(']"', ']').replace('],"', ']","')
            return len(json.loads(corrected_x))
        except:
            return 0

# Function to remove outliers
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3 - q1
    fence_low = q1 - 2 * iqr
    fence_high = q3 + 2 * iqr
    df_out = df_in.loc[(df_in[col_name] <= fence_high) & (df_in[col_name] >= fence_low)]
    return df_out

# Function to load and process data
@st.cache_data
def load_and_process_data(data_file, interaction_file, pics_file):
    data = pd.read_csv(data_file, parse_dates=['activation_date'], dayfirst=True)
    interaction = pd.read_csv(interaction_file, parse_dates=['request_date'], dayfirst=True)
    pics = pd.read_csv(pics_file, sep='\t')

    # Data processing steps
    interaction['request_date'] = pd.to_datetime(interaction['request_date'], errors='coerce')
    pics['photo_count'] = pics['photo_urls'].apply(correction)
    pics.drop('photo_urls', axis=1, inplace=True)

    num_req = pd.merge(data, interaction, on='property_id')[['property_id', 'request_date', 'activation_date']]
    num_req['request_day'] = (num_req['request_date'] - num_req['activation_date']).dt.days

    num_req_within_3d = num_req[num_req['request_day'] < 3].groupby('property_id').agg({'request_day': 'count'}).reset_index()
    num_req_within_3d.rename({'request_day': 'request_day_within_3d'}, axis=1, inplace=True)

    num_req_within_7d = num_req[num_req['request_day'] < 7].groupby('property_id').agg({'request_day': 'count'}).reset_index()
    num_req_within_7d.rename({'request_day': 'request_day_within_7d'}, axis=1, inplace=True)

    def divide(x):
        if x in [1, 2]:
            return 'cat_1_to_2'
        elif x in [3, 4, 5]:
            return 'cat_3_to_5'
        else:
            return 'cat_above_5'

    num_req_within_3d['categories_3day'] = num_req_within_3d['request_day_within_3d'].apply(divide)
    num_req_within_7d['categories_7day'] = num_req_within_7d['request_day_within_7d'].apply(divide)

    label_data = pd.merge(num_req_within_7d, num_req_within_3d, on='property_id', how='left')
    data_with_pics = pd.merge(data, pics, on='property_id', how='left')
    dataset = pd.merge(data_with_pics, label_data, on='property_id')
    dataset.drop(['property_id', 'activation_date', 'latitude', 'longitude', 'pin_code', 'locality'], axis=1, inplace=True)
    
    return data, interaction, pics, dataset

# Function to prepare data for modeling
def prepare_modeling_data(df, category_choice):
    # Remove outliers
    df_clean = df.copy()
    for col in ['property_age', 'property_size', 'rent', 'deposit', 'photo_count']:
        df_clean = remove_outlier(df_clean, col)

    # Capping functions
    def capping_for_3days(x):
        num = 10
        return min(x, num)

    def capping_for_7days(x):
        num = 20
        return min(x, num)

    df_clean['request_day_within_3d_capping'] = df_clean['request_day_within_3d'].apply(capping_for_3days)
    df_clean['request_day_within_7d_capping'] = df_clean['request_day_within_7d'].apply(capping_for_7days)

    # One-Hot Encoding
    X = df_clean.drop(['request_day_within_7d', 'categories_7day', 'request_day_within_3d',
                      'categories_3day', 'request_day_within_3d_capping',
                      'request_day_within_7d_capping'], axis=1)
    x_cat_withNull = df_clean[X.select_dtypes(include=['O']).columns]
    x_remain_withNull = df_clean[X.select_dtypes(exclude=['O']).columns]
    y = df_clean[['request_day_within_7d', 'categories_7day', 'request_day_within_3d',
                'categories_3day', 'request_day_within_3d_capping',
                'request_day_within_7d_capping']]

    # Handle null values
    x_remain = x_remain_withNull.fillna(x_remain_withNull.mean())
    x_cat = x_cat_withNull.apply(lambda col: col.fillna(col.mode()[0]))

    # One-Hot Encoding
    ohe = OneHotEncoder(categories='auto', handle_unknown='ignore', sparse_output=False)
    feature_train = ohe.fit_transform(x_cat)
    feature_labels = ohe.get_feature_names_out(x_cat.columns)
    df_features = pd.DataFrame(feature_train, columns=feature_labels)

    # Scaling
    sc = MinMaxScaler()
    x_remain_scaled = sc.fit_transform(x_remain)
    x_remain_scaled = pd.DataFrame(x_remain_scaled, columns=x_remain.columns)

    if category_choice == '3 Days':
        data_with_days = pd.concat([df_features.reset_index(drop=True),
                                    x_remain_scaled.reset_index(drop=True),
                                    y[['request_day_within_3d',
                                       'request_day_within_3d_capping',
                                       'categories_3day']].reset_index(drop=True)], axis=1)
        target_variable = 'request_day_within_3d_capping'
    else:
        data_with_days = pd.concat([df_features.reset_index(drop=True),
                                    x_remain_scaled.reset_index(drop=True),
                                    y[['request_day_within_7d',
                                       'request_day_within_7d_capping',
                                       'categories_7day']].reset_index(drop=True)], axis=1)
        target_variable = 'request_day_within_7d_capping'

    data_with_days.dropna(inplace=True)

    return data_with_days, target_variable

# Introduction Section
if selection == "Introduction":
    st.title("Property Data Analysis")
    st.markdown("""
    Welcome to the **Property Data Analysis** app! This application allows you to explore property data, visualize trends, and evaluate predictive models.

    **Features:**
    - Upload your own datasets.
    - Interactive visualizations with customizable parameters.
    - Model training and evaluation with adjustable settings.

    **Instructions:**
    - Use the sidebar to navigate between sections.
    - Download the required datasets from the GitHub link in the sidebar.
    - Upload the datasets in the respective sections to proceed.
    """)

# Data Upload Section
elif selection == "Data Upload":
    st.title("Data Upload")
    st.markdown("""
    Use the sidebar to upload the required data files:
    - `property_data_set.csv`
    - `property_interactions.csv`
    - `property_photos.tsv`

    Ensure all files are uploaded to proceed.
    """)

    # Sidebar Upload Data Files
    with st.sidebar.expander("📤 Upload Data Files"):
        data_file = st.file_uploader("Upload `property_data_set.csv`", type=["csv"])
        interaction_file = st.file_uploader("Upload `property_interactions.csv`", type=["csv"])
        pics_file = st.file_uploader("Upload `property_photos.tsv`", type=["tsv"])

    if data_file and interaction_file and pics_file:
        st.success("All files uploaded successfully.")
        # Optionally, display a preview of the data
        if st.checkbox("Show Data Preview"):
            data, interaction, pics, dataset = load_and_process_data(data_file, interaction_file, pics_file)
            st.subheader("Property Data")
            st.dataframe(data.head())
            st.subheader("Interaction Data")
            st.dataframe(interaction.head())
            st.subheader("Photos Data")
            st.dataframe(pics.head())
    else:
        st.warning("Please upload all three data files to proceed.")

# Data Information Section
elif selection == "Data Information":
    st.title("Data Information")
    st.markdown("""
    Upload the required data files to view detailed information about each dataset, including data types, missing values, and any inconsistencies.
    """)

    # Sidebar Upload Data Files
    with st.sidebar.expander("📤 Upload Data Files"):
        data_file = st.file_uploader("Upload `property_data_set.csv`", type=["csv"])
        interaction_file = st.file_uploader("Upload `property_interactions.csv`", type=["csv"])
        pics_file = st.file_uploader("Upload `property_photos.tsv`", type=["tsv"])

    if data_file and interaction_file and pics_file:
        # Load the data files
        data, interaction, pics, dataset = load_and_process_data(data_file, interaction_file, pics_file)

        # Interaction Data Info
        st.header("Interaction Data Info")
        buffer = io.StringIO()
        interaction.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        # Check for invalid dates
        invalid_dates = interaction[interaction['request_date'].isna()]
        st.subheader("Invalid Dates in Interaction Data")
        st.write(invalid_dates)

        # Number of missing values
        st.subheader("Missing Values in Interaction Data")
        st.write(interaction.isna().sum())

        # Pics Data Info
        st.header("Pics Data Info")
        buffer = io.StringIO()
        pics.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader("Missing Values in Pics Data")
        st.write(pics.isna().sum())

        # Property Data Info
        st.header("Property Data Info")
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader("Missing Values in Property Data")
        st.write(data.isna().sum())

    else:
        st.warning("Please upload all three data files to proceed.")

# Data Visualization Section
elif selection == "Data Visualization":
    st.title("Data Visualization")
    
    # Sidebar Options for Data Visualization
    # Category Selection and Visualization Parameters
    with st.sidebar:
        st.markdown("### 📊 **Visualization Options**")
        selected_features = st.multiselect(
            "Select Numeric Features to Plot",
            options=['property_age', 'property_size', 'rent', 'deposit', 'photo_count'],
            default=['property_age', 'property_size']
        )

        category_choice = st.radio(
            "Select Category for Analysis",
            options=['3 Days', '7 Days']
        )

    # Sidebar Upload Data Files
    with st.sidebar.expander("📤 Upload Data Files"):
        data_file = st.file_uploader("Upload `property_data_set.csv`", type=["csv"])
        interaction_file = st.file_uploader("Upload `property_interactions.csv`", type=["csv"])
        pics_file = st.file_uploader("Upload `property_photos.tsv`", type=["tsv"])

    if data_file and interaction_file and pics_file:
        # Load and process data
        data, interaction, pics, dataset = load_and_process_data(data_file, interaction_file, pics_file)

        # Categorical variable selection
        categorical_vars = dataset.select_dtypes(include=['object']).columns.tolist()
        if categorical_vars:
            selected_categorical = st.sidebar.selectbox(
                "Select Categorical Variable for Count Plot",
                options=categorical_vars
            )
        else:
            st.warning("No categorical variables found in the dataset.")
            selected_categorical = None

        # Visualization based on category choice
        if category_choice == '3 Days':
            st.subheader("Histogram of Number of Requests in First 3 Days")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(dataset, x="request_day_within_3d", ax=ax, kde=True)
            ax.set_title('Histogram of Number of Requests in First 3 Days')
            st.pyplot(fig)

            st.subheader("Value Counts for Categories Within 3 Days")
            fig, ax = plt.subplots()
            sns.countplot(y=dataset['categories_3day'], ax=ax, palette="viridis")
            ax.set_title('Value Count for Each Category Within 3 Days')
            st.pyplot(fig)

            if selected_features:
                st.subheader("Pairplot of Selected Features vs. Requests Within 3 Days")
                sns.pairplot(data=dataset,
                             vars=selected_features + ['request_day_within_3d'],
                             diag_kind='kde')
                st.pyplot()
            else:
                st.warning("Please select at least one feature to display the pairplot.")

        else:
            st.subheader("Histogram of Number of Requests in First 7 Days")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(dataset, x="request_day_within_7d", ax=ax, kde=True)
            ax.set_title('Histogram of Number of Requests in First 7 Days')
            st.pyplot(fig)

            st.subheader("Value Counts for Categories Within 7 Days")
            fig, ax = plt.subplots()
            sns.countplot(y=dataset['categories_7day'], ax=ax, palette="magma")
            ax.set_title('Value Count for Each Category Within 7 Days')
            st.pyplot(fig)

            if selected_features:
                st.subheader("Pairplot of Selected Features vs. Requests Within 7 Days")
                sns.pairplot(data=dataset,
                             vars=selected_features + ['request_day_within_7d'],
                             diag_kind='kde')
                st.pyplot()
            else:
                st.warning("Please select at least one feature to display the pairplot.")

        # Interactive scatter matrix
        st.subheader("Interactive Scatter Matrix")
        columns_to_plot = selected_features.copy()
        if category_choice == '3 Days':
            columns_to_plot.append('request_day_within_3d')
        else:
            columns_to_plot.append('request_day_within_7d')

        if len(columns_to_plot) > 1:
            fig = px.scatter_matrix(dataset[columns_to_plot],
                                    dimensions=columns_to_plot,
                                    title="Scatter Matrix for Selected Features",
                                    template="simple_white",
                                    height=800, width=800)
            st.plotly_chart(fig)
        else:
            st.warning("Please select at least two features to display the scatter matrix.")

        # Correlation Heatmap with Options
        st.subheader("Correlation Heatmap")
        heatmap_features = st.multiselect(
            "Select Features for Heatmap",
            options=dataset.select_dtypes(include=[np.number]).columns.tolist(),
            default=dataset.select_dtypes(include=[np.number]).columns.tolist()
        )

        if len(heatmap_features) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(dataset[heatmap_features].corr(), annot=True, cmap="YlGnBu", linewidths=0.5)
            st.pyplot()
        else:
            st.warning("Please select at least two features for the heatmap.")

        # Count Plot for Selected Categorical Variable
        if selected_categorical:
            st.subheader(f"Count Plot for {selected_categorical}")
            fig, ax = plt.subplots()
            sns.countplot(y=dataset[selected_categorical], order=dataset[selected_categorical].value_counts().index, ax=ax, palette="coolwarm")
            ax.set_title(f'Count Plot for {selected_categorical}')
            st.pyplot(fig)

    else:
        st.warning("Please upload all three data files to proceed.")

# Model Training and Evaluation Section
elif selection == "Model Training and Evaluation":
    st.title("Model Training and Evaluation")
    
    # Sidebar Options for Modeling
    with st.sidebar:
        st.markdown("### 🤖 **Model Options**")
        category_choice = st.radio(
            "Select Category for Analysis",
            options=['3 Days', '7 Days']
        )
        model_type = st.selectbox("Select Model Type", options=['Linear Regression', 'KNN Regressor'])

        if model_type == 'KNN Regressor':
            n_neighbors = st.slider("Select Number of Neighbors (K)", min_value=1, max_value=20, value=5)
        else:
            n_neighbors = None

    # Sidebar Upload Data Files
    with st.sidebar.expander("📤 Upload Data Files"):
        data_file = st.file_uploader("Upload `property_data_set.csv`", type=["csv"])
        interaction_file = st.file_uploader("Upload `property_interactions.csv`", type=["csv"])
        pics_file = st.file_uploader("Upload `property_photos.tsv`", type=["tsv"])

    if data_file and interaction_file and pics_file:
        # Load and process data
        data, interaction, pics, dataset = load_and_process_data(data_file, interaction_file, pics_file)

        # Prepare data for modeling
        data_with_days, target_variable = prepare_modeling_data(dataset, category_choice)

        # Display target variable
        st.write(f"**Target Variable:** {target_variable}")

        # Model Training and Evaluation
        X_days = data_with_days.drop(['request_day_within_3d',
                                      'request_day_within_3d_capping',
                                      'categories_3day',
                                      'request_day_within_7d',
                                      'request_day_within_7d_capping',
                                      'categories_7day'], axis=1, errors='ignore')
        y_days = data_with_days[target_variable]
        seed = 42
        X_train, X_test, y_train, y_test = train_test_split(X_days, y_days, test_size=0.2, random_state=seed)

        if model_type == 'Linear Regression':
            model = LinearRegression()
        elif model_type == 'KNN Regressor':
            model = KNeighborsRegressor(n_neighbors=n_neighbors)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Manual Mean Squared Error
        def manual_mean_squared_error(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        mse = manual_mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        st.write(f"**{model_type}: RMSE = {rmse:.4f}**")

        # Show actual vs predicted
        st.subheader("Actual vs Predicted Values")
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        st.write(results_df.head(10))

        # Optional: Plot Actual vs Predicted
        st.subheader("Actual vs Predicted Scatter Plot")
        fig, ax = plt.subplots()
        sns.scatterplot(x='Actual', y='Predicted', data=results_df, ax=ax, color='teal')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
        ax.set_title('Actual vs Predicted')
        st.pyplot(fig)

    else:
        st.warning("Please upload all three data files to proceed.")

# Handle Invalid Selection
else:
    st.warning("Invalid selection.")
