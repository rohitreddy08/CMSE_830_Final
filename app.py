import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import io

# Set up the page configuration
st.set_page_config(page_title="Property Data Analysis", layout="wide")

# Basic documentation
st.title("Property Data Analysis")
st.markdown("""
Welcome to the **Property Data Analysis** app! This application allows you to explore property data, visualize trends, and evaluate predictive models.

**Features:**
- Upload your own datasets.
- Interactive visualizations with customizable parameters.
- Model training and evaluation with adjustable settings.

**Instructions:**
- Use the sidebar to upload data files.
- Choose visualization options to customize the plots.
- Select and configure models to train and evaluate.
""")

# Sidebar for file uploads
st.sidebar.title("Upload Data Files")
data_file = st.sidebar.file_uploader("Upload `property_data_set.csv`", type=["csv"])
interaction_file = st.sidebar.file_uploader("Upload `property_interactions.csv`", type=["csv"])
pics_file = st.sidebar.file_uploader("Upload `property_photos.tsv`", type=["tsv"])

if data_file and interaction_file and pics_file:
    # Load the data files
    data = pd.read_csv(data_file, parse_dates=['activation_date'], dayfirst=True)
    interaction = pd.read_csv(interaction_file, parse_dates=['request_date'], dayfirst=True)
    pics = pd.read_csv(pics_file, sep='\t')

    # Interaction Data Info
    st.header("Interaction Data Info")
    buffer = io.StringIO()
    interaction.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Re-parse request_date to check for invalid dates
    interaction['request_date'] = pd.to_datetime(interaction['request_date'], errors='coerce')

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

    # Apply the correction function
    pics['photo_count'] = pics['photo_urls'].apply(correction)
    pics.drop('photo_urls', axis=1, inplace=True)
    st.subheader("Pics Data After Processing")
    st.write(pics.head())

    # Merge data with interactions on property_id
    num_req = pd.merge(data, interaction, on='property_id')[['property_id', 'request_date', 'activation_date']]
    st.subheader("Merged Data Sample")
    st.write(num_req.head())

    # Calculate the difference in days
    num_req['request_day'] = (num_req['request_date'] - num_req['activation_date']).dt.days
    st.subheader("Difference in Days Between Request and Activation Dates")
    st.write(num_req[['request_date', 'activation_date', 'request_day']].head())

    # Requests within 3 days
    num_req_within_3d = num_req[num_req['request_day'] < 3].groupby('property_id').agg({'request_day': 'count'}).reset_index()
    num_req_within_3d.rename({'request_day': 'request_day_within_3d'}, axis=1, inplace=True)
    st.subheader("Number of Requests Within 3 Days")
    st.write(num_req_within_3d.head())

    # Categorize the number of requests
    def divide(x):
        if x in [1, 2]:
            return 'cat_1_to_2'
        elif x in [3, 4, 5]:
            return 'cat_3_to_5'
        else:
            return 'cat_above_5'

    num_req_within_3d['categories_3day'] = num_req_within_3d['request_day_within_3d'].apply(divide)
    st.subheader("Categories Within 3 Days")
    st.write(num_req_within_3d['categories_3day'].value_counts())

    # Requests within 7 days
    num_req_within_7d = num_req[num_req['request_day'] < 7].groupby('property_id').agg({'request_day': 'count'}).reset_index()
    num_req_within_7d.rename({'request_day': 'request_day_within_7d'}, axis=1, inplace=True)
    num_req_within_7d['categories_7day'] = num_req_within_7d['request_day_within_7d'].apply(divide)
    st.subheader("Number of Requests Within 7 Days")
    st.write(num_req_within_7d.head())
    st.subheader("Categories Within 7 Days")
    st.write(num_req_within_7d['categories_7day'].value_counts())

    # Merge label data
    label_data = pd.merge(num_req_within_7d, num_req_within_3d, on='property_id', how='left')
    st.subheader("Label Data")
    st.write(label_data.head())

    # Merge data with pics
    data_with_pics = pd.merge(data, pics, on='property_id', how='left')
    st.subheader("Data with Pics")
    st.write(data_with_pics.head())

    # Final dataset
    dataset = pd.merge(data_with_pics, label_data, on='property_id')
    st.subheader("Final Dataset")
    st.write(dataset.head())

    # Drop unnecessary columns
    dataset.drop(['property_id', 'activation_date', 'latitude', 'longitude', 'pin_code', 'locality'], axis=1, inplace=True)

    # Display missing values
    st.subheader("Missing Values in Final Dataset")
    st.write(dataset.isna().sum())

    # Interactive element: Select features to visualize
    st.sidebar.title("Visualization Options")
    selected_features = st.sidebar.multiselect(
        "Select Numeric Features to Plot",
        options=['property_age', 'property_size', 'rent', 'deposit', 'photo_count'],
        default=['property_age', 'property_size']
    )

    # Interactive element: Choose the category for analysis
    category_choice = st.sidebar.radio(
        "Select Category for Analysis",
        options=['3 Days', '7 Days']
    )

    # Histogram of requests
    if category_choice == '3 Days':
        st.subheader("Histogram of Number of Requests in First 3 Days")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(dataset, x="request_day_within_3d", ax=ax)
        ax.set_title('Histogram of Number of Requests in First 3 Days')
        st.pyplot(fig)

        # Value counts for categories
        st.subheader("Value Counts for Categories Within 3 Days")
        fig, ax = plt.subplots()
        sns.countplot(y=dataset['categories_3day'], ax=ax)
        ax.set_title('Value Count for Each Category Within 3 Days')
        st.pyplot(fig)

        # Pairplot for selected features vs. requests within 3 days
        if selected_features:
            st.subheader("Pairplot of Selected Features vs. Requests Within 3 Days")
            sns.pairplot(data=dataset,
                         x_vars=selected_features,
                         y_vars=['request_day_within_3d'])
            st.pyplot()
        else:
            st.warning("Please select at least one feature to display the pairplot.")

    else:
        st.subheader("Histogram of Number of Requests in First 7 Days")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(dataset, x="request_day_within_7d", ax=ax)
        ax.set_title('Histogram of Number of Requests in First 7 Days')
        st.pyplot(fig)

        # Value counts for categories
        st.subheader("Value Counts for Categories Within 7 Days")
        fig, ax = plt.subplots()
        sns.countplot(y=dataset['categories_7day'], ax=ax)
        ax.set_title('Value Count for Each Category Within 7 Days')
        st.pyplot(fig)

        # Pairplot for selected features vs. requests within 7 days
        if selected_features:
            st.subheader("Pairplot of Selected Features vs. Requests Within 7 Days")
            sns.pairplot(data=dataset,
                         x_vars=selected_features,
                         y_vars=['request_day_within_7d'])
            st.pyplot()
        else:
            st.warning("Please select at least one feature to display the pairplot.")

    # Interactive scatter matrix
    st.subheader("Interactive Scatter Matrix")
    columns_to_plot = selected_features
    if category_choice == '3 Days':
        columns_to_plot += ['request_day_within_3d']
    else:
        columns_to_plot += ['request_day_within_7d']

    if len(columns_to_plot) > 1:
        fig = px.scatter_matrix(dataset[columns_to_plot],
                                dimensions=columns_to_plot,
                                title="Scatter Matrix for Selected Features",
                                template="simple_white",
                                height=800, width=800)
        st.plotly_chart(fig)
    else:
        st.warning("Please select at least two features to display the scatter matrix.")

    # Categorical and numeric columns
    df_cat = dataset.select_dtypes(include=['object'])
    df_num = dataset.select_dtypes(exclude=['object'])

    st.subheader("Categorical Columns")
    st.write(list(df_cat.columns))
    st.subheader("Numeric Columns")
    st.write(list(df_num.columns))

    # Value counts in categorical columns
    for col in df_cat.columns[:-2]:
        st.write(f"Value Counts for {col}")
        st.write(df_cat[col].value_counts())

    # Count plots for categorical columns
    for col in df_cat.columns[:-2]:
        fig, ax = plt.subplots()
        sns.countplot(x=col, data=dataset, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Box plots for numeric columns
    st.subheader("Box Plots for Numeric Features")
    num_cols = df_num.columns.tolist()
    num_columns_per_row = 5
    num_rows = len(num_cols) // num_columns_per_row + 1

    fig = sp.make_subplots(
        rows=num_rows,
        cols=num_columns_per_row,
        shared_yaxes=False,
        subplot_titles=num_cols
    )

    for i, col in enumerate(num_cols):
        row = i // num_columns_per_row + 1
        col_num = i % num_columns_per_row + 1
        fig.add_trace(go.Box(y=df_num[col], name=col, boxpoints='outliers'), row=row, col=col_num)

    fig.update_layout(
        height=400 * num_rows,
        width=250 * num_columns_per_row,
        showlegend=False,
        title_text="Interactive Box Plots for Numeric Features",
        template="plotly_dark",
        xaxis_tickangle=45,
        margin=dict(l=50, r=50, b=100, t=100)
    )

    st.plotly_chart(fig)

    # Remove outliers
    def remove_outlier(df_in, col_name):
        q1 = df_in[col_name].quantile(0.25)
        q3 = df_in[col_name].quantile(0.75)
        iqr = q3 - q1
        fence_low = q1 - 2 * iqr
        fence_high = q3 + 2 * iqr
        df_out = df_in.loc[(df_in[col_name] <= fence_high) & (df_in[col_name] >= fence_low)]
        return df_out

    df = dataset.copy()
    for col in selected_features:
        df = remove_outlier(df, col)

    # Capping functions
    def capping_for_3days(x):
        num = 10
        return min(x, num)

    def capping_for_7days(x):
        num = 20
        return min(x, num)

    df['request_day_within_3d_capping'] = df['request_day_within_3d'].apply(capping_for_3days)
    df['request_day_within_7d_capping'] = df['request_day_within_7d'].apply(capping_for_7days)

    # Correlation heatmap without annotations
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), cmap="YlGnBu", annot=False)  # Remove annotations for readability
    plt.title('Correlation Heatmap Without Values')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot()

    # One-Hot Encoding
    X = df.drop(['request_day_within_7d', 'categories_7day', 'request_day_within_3d',
                 'categories_3day', 'request_day_within_3d_capping',
                 'request_day_within_7d_capping'], axis=1)
    x_cat_withNull = df[X.select_dtypes(include=['O']).columns]
    x_remain_withNull = df[X.select_dtypes(exclude=['O']).columns]
    y = df[['request_day_within_7d', 'categories_7day', 'request_day_within_3d',
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
    st.subheader("Features After One-Hot Encoding")
    st.write(df_features.head())

    # Scaling
    sc = MinMaxScaler()
    x_remain_scaled = sc.fit_transform(x_remain)
    x_remain_scaled = pd.DataFrame(x_remain_scaled, columns=x_remain.columns)

    # Concatenate data
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
    st.write(f"Shape of Data: {data_with_days.shape}")

    # Correlation heatmap for data_with_days without annotations
    st.subheader("Correlation Heatmap for Data")
    numeric_columns_days = data_with_days.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_with_days[numeric_columns_days].corr(), annot=False, cmap="YlGnBu", linewidths=0.5)  # annot=False
    st.pyplot()

    # Model Training and Evaluation
    st.header("Model Training and Evaluation")
    X_days = data_with_days.drop(['request_day_within_3d',
                                  'request_day_within_3d_capping',
                                  'categories_3day',
                                  'request_day_within_7d',
                                  'request_day_within_7d_capping',
                                  'categories_7day'], axis=1, errors='ignore')
    y_days = data_with_days[target_variable]
    seed = 42
    X_train, X_test, y_train, y_test = train_test_split(X_days, y_days, test_size=0.2, random_state=seed)

    # Interactive element: Select model type
    model_type = st.selectbox("Select Model Type", options=['Linear Regression', 'KNN Regressor'])

    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'KNN Regressor':
        n_neighbors = st.slider("Select Number of Neighbors (K)", min_value=1, max_value=20, value=5)
        model = KNeighborsRegressor(n_neighbors=n_neighbors)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rms = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f"**{model_type}: RMSE = {rms:.4f}**")

    # Show actual vs predicted
    st.subheader("Actual vs Predicted Values")
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.write(results_df.head(10))

else:
    st.warning("Please upload all three data files to proceed.")
