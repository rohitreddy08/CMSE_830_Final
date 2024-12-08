import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import plotly.express as px
import altair as alt
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
    "Data Exploration",
    "Data Cleaning",
    "Data Visualization",
    "Model Training and Evaluation"
])

# Add GitHub link below navigation
st.sidebar.markdown("""
---
### 📂 **Download Data Files**
[GitHub Repository](https://github.com/rohitreddy08/CMSE_830_Final/tree/main/Data)
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
    fence_low = q1 - 2 * iqr  # Using 2*IQR for this application
    fence_high = q3 + 2 * iqr
    original_count = df_in.shape[0]
    df_out = df_in.loc[(df_in[col_name] <= fence_high) & (df_in[col_name] >= fence_low)]
    cleaned_count = df_out.shape[0]
    return df_out, original_count - cleaned_count

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

# Function to prepare data for modeling and track cleaning steps
def prepare_modeling_data(df, category_choice):
    cleaning_steps = {
        'outlier_removal': {},
        'imputations': {},
        'encodings': [],
        'scaling': []
    }
    
    # Remove outliers
    df_clean = df.copy()
    outlier_counts = {}
    for col in ['property_age', 'property_size', 'rent', 'deposit', 'photo_count']:
        df_clean, removed = remove_outlier(df_clean, col)
        outlier_counts[col] = removed
    cleaning_steps['outlier_removal'] = outlier_counts

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
    missing_before = x_remain_withNull.isna().sum()
    x_remain = x_remain_withNull.fillna(x_remain_withNull.mean())
    missing_after = x_remain.isna().sum()
    imputed_cols_num = missing_before[missing_before > 0].index.tolist()
    cleaning_steps['imputations']['numerical'] = {
        'columns': imputed_cols_num,
        'method': 'Mean Imputation'
    }

    x_cat_missing_before = x_cat_withNull.isna().sum()
    x_cat = x_cat_withNull.apply(lambda col: col.fillna(col.mode()[0]))
    x_cat_missing_after = x_cat.isna().sum()
    imputed_cols_cat = x_cat_missing_before[x_cat_missing_before > 0].index.tolist()
    cleaning_steps['imputations']['categorical'] = {
        'columns': imputed_cols_cat,
        'method': 'Mode Imputation'
    }

    # One-Hot Encoding
    ohe = OneHotEncoder(categories='auto', handle_unknown='ignore', sparse_output=False)
    feature_train = ohe.fit_transform(x_cat)
    feature_labels = ohe.get_feature_names_out(x_cat.columns)
    df_features = pd.DataFrame(feature_train, columns=feature_labels)
    encoded_columns = x_cat.columns.tolist()
    cleaning_steps['encodings'] = encoded_columns

    # Scaling
    sc = MinMaxScaler()
    x_remain_scaled = sc.fit_transform(x_remain)
    x_remain_scaled = pd.DataFrame(x_remain_scaled, columns=x_remain.columns)
    scaled_columns = x_remain.columns.tolist()
    cleaning_steps['scaling'] = scaled_columns

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

    return data_with_days, target_variable, cleaning_steps

# Introduction Section
if selection == "Introduction":
    st.title("🏠 Property Data Analysis Dashboard")
    
    st.markdown("""
    ### Welcome to the Property Data Analysis Dashboard!
    
    This application is designed to help you **explore**, **visualize**, and **model** property-related data seamlessly. Whether you're a data enthusiast, a real estate analyst, or a developer, this tool offers a comprehensive suite of features to assist you in deriving meaningful insights from your property datasets.
    
    ---
    
    ### 📈 **App Features**
    
    1. **Data Upload**
        - **What:** Easily upload your property datasets.
        - **Files Required:**
            - `property_data_set.csv`: Contains detailed information about properties.
            - `property_interactions.csv`: Logs interactions or requests related to properties.
            - `property_photos.tsv`: Contains URLs or metadata for property photos.
    
    2. **Data Exploration**
        - **What:** Dive deep into each dataset to understand its structure and contents.
        - **Includes:**
            - Sample data previews.
            - Data types and shapes.
            - Statistical summaries.
            - Insights into missing or invalid data.
    
    3. **Data Cleaning**
        - **What:** Prepare your data for analysis and modeling by addressing inconsistencies.
        - **Processes Involved:**
            - Outlier removal using the Interquartile Range (IQR) method.
            - Handling missing values through mean and mode imputation.
            - Encoding categorical variables for machine learning compatibility.
            - Feature scaling to normalize numerical data.
    
    4. **Data Visualization**
        - **What:** Create interactive and insightful visualizations to uncover trends and patterns.
        - **Tools & Plots:**
            - Histograms and count plots.
            - Pairplots and scatter matrices.
            - Correlation heatmaps.
            - Interactive charts using Plotly and Altair.
    
    5. **Model Training and Evaluation**
        - **What:** Build and assess predictive models based on your cleaned data.
        - **Models Available:**
            - **Linear Regression:** Understand relationships between variables.
            - **K-Nearest Neighbors (KNN) Regressor:** Capture non-linear patterns.
        - **Evaluation Metrics:**
            - Root Mean Squared Error (RMSE).
            - Visual comparisons of actual vs. predicted values.
    
    ---
    
    ### 🛠 **How to Get Started**
    
    1. **Download Data Files:**
        - Access the required datasets from the [GitHub Repository](https://github.com/rohitreddy08/CMSE_830_Final/tree/main/Data).
    
    2. **Upload Your Data:**
        - Navigate to the **Data Upload** section using the sidebar.
        - Upload the `property_data_set.csv`, `property_interactions.csv`, and `property_photos.tsv` files.
    
    3. **Explore and Clean Your Data:**
        - Proceed to the **Data Exploration** and **Data Cleaning** sections to understand and prepare your data.
    
    4. **Visualize Insights:**
        - Use the **Data Visualization** section to create charts and plots that highlight key trends.
    
    5. **Train and Evaluate Models:**
        - Head over to the **Model Training and Evaluation** section to build predictive models and assess their performance.
    
    ---
    
    ### 📚 **Additional Resources**
    
    - **Support:** For any queries or issues, please contact [n.rohitreddy08@gmail.com](mailto:n.rohitreddy08@gmail.com).
    
    ---
    
    **Enjoy exploring your property data!**
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

# Data Exploration Section
elif selection == "Data Exploration":
    st.title("Data Exploration")
    st.markdown("""
    Upload the required data files to explore each dataset in detail, including sample data, data types, shapes, and statistical summaries.
    """)

    # Sidebar Upload Data Files
    with st.sidebar.expander("📤 Upload Data Files"):
        data_file = st.file_uploader("Upload `property_data_set.csv`", type=["csv"])
        interaction_file = st.file_uploader("Upload `property_interactions.csv`", type=["csv"])
        pics_file = st.file_uploader("Upload `property_photos.tsv`", type=["tsv"])

    if data_file and interaction_file and pics_file:
        # Load the data files
        data, interaction, pics, dataset = load_and_process_data(data_file, interaction_file, pics_file)

        st.header("Property Data Exploration")
        with st.expander("🔍 View Property Data"):
            st.subheader("Sample Data")
            st.dataframe(data.head())

            st.subheader("Data Types")
            data_types = pd.DataFrame(data.dtypes, columns=['Data Type'])
            st.table(data_types)

            st.subheader("Shape of Dataset")
            st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

            st.subheader("Statistical Summary")
            st.write(data.describe(include='all').transpose())

        st.header("Interaction Data Exploration")
        with st.expander("🔍 View Interaction Data"):
            st.subheader("Sample Data")
            st.dataframe(interaction.head())

            st.subheader("Data Types")
            interaction_types = pd.DataFrame(interaction.dtypes, columns=['Data Type'])
            st.table(interaction_types)

            st.subheader("Shape of Dataset")
            st.write(f"Rows: {interaction.shape[0]}, Columns: {interaction.shape[1]}")

            st.subheader("Statistical Summary")
            st.write(interaction.describe(include='all').transpose())

            st.subheader("Invalid Dates in Interaction Data")
            invalid_dates = interaction[interaction['request_date'].isna()]
            st.write(invalid_dates)

            st.subheader("Missing Values")
            st.write(interaction.isna().sum())

        st.header("Pics Data Exploration")
        with st.expander("🔍 View Pics Data"):
            st.subheader("Sample Data")
            st.dataframe(pics.head())

            st.subheader("Data Types")
            pics_types = pd.DataFrame(pics.dtypes, columns=['Data Type'])
            st.table(pics_types)

            st.subheader("Shape of Dataset")
            st.write(f"Rows: {pics.shape[0]}, Columns: {pics.shape[1]}")

            st.subheader("Statistical Summary")
            st.write(pics.describe(include='all').transpose())

            st.subheader("Missing Values")
            st.write(pics.isna().sum())

    else:
        st.warning("Please upload all three data files to proceed.")

# Data Cleaning Section
elif selection == "Data Cleaning":
    st.title("Data Cleaning")
    st.markdown("""
    This section provides detailed information about the data cleaning steps performed, including outlier removals, imputations, encodings, and scaling.
    """)

    # Sidebar Upload Data Files
    with st.sidebar.expander("📤 Upload Data Files"):
        data_file = st.file_uploader("Upload `property_data_set.csv`", type=["csv"])
        interaction_file = st.file_uploader("Upload `property_interactions.csv`", type=["csv"])
        pics_file = st.file_uploader("Upload `property_photos.tsv`", type=["tsv"])

    if data_file and interaction_file and pics_file:
        # Load the data files
        data, interaction, pics, dataset = load_and_process_data(data_file, interaction_file, pics_file)

        # Prepare data for modeling and get cleaning steps
        # Allow user to select category_choice dynamically
        category_choice = st.radio(
            "Select Category for Data Cleaning",
            options=['3 Days', '7 Days'],
            index=0
        )
        data_with_days, target_variable, cleaning_steps = prepare_modeling_data(dataset, category_choice)

        st.header("Data Cleaning Steps Overview")

        # 1. Outlier Removal
        st.subheader("1. Outlier Removal")
        outlier_df = pd.DataFrame({
            'Column': list(cleaning_steps['outlier_removal'].keys()),
            'Outliers Removed': list(cleaning_steps['outlier_removal'].values())
        })
        st.table(outlier_df)
        st.markdown("""
        **Methodology:** Outliers were removed based on the Interquartile Range (IQR) method. Specifically, values outside the range of `Q1 - 2*IQR` and `Q3 + 2*IQR` were considered outliers and excluded from the dataset.
        """)

        # 2. Missing Value Imputations
        st.subheader("2. Missing Value Imputations")
        # Numerical Imputations
        num_imputations = cleaning_steps['imputations']['numerical']
        if num_imputations['columns']:
            st.markdown("**Numerical Columns:**")
            num_impute_df = pd.DataFrame({
                'Column': num_imputations['columns'],
                'Imputation Method': [num_imputations['method']] * len(num_imputations['columns'])
            })
            st.table(num_impute_df)
        else:
            st.markdown("**Numerical Columns:** No missing values detected.")

        # Categorical Imputations
        cat_imputations = cleaning_steps['imputations']['categorical']
        if cat_imputations['columns']:
            st.markdown("**Categorical Columns:**")
            cat_impute_df = pd.DataFrame({
                'Column': cat_imputations['columns'],
                'Imputation Method': [cat_imputations['method']] * len(cat_imputations['columns'])
            })
            st.table(cat_impute_df)
        else:
            st.markdown("**Categorical Columns:** No missing values detected.")

        # 3. Categorical Variable Encodings
        st.subheader("3. Categorical Variable Encodings")
        if cleaning_steps['encodings']:
            st.markdown("**One-Hot Encoded Columns:**")
            st.write(cleaning_steps['encodings'])
        else:
            st.markdown("**One-Hot Encoded Columns:** None.")

        # 4. Feature Scaling
        st.subheader("4. Feature Scaling")
        if cleaning_steps['scaling']:
            st.markdown("**Scaled Numerical Columns (Min-Max Scaling):**")
            st.write(cleaning_steps['scaling'])
        else:
            st.markdown("**Scaled Numerical Columns (Min-Max Scaling):** None.")

        # Summary
        st.header("Summary of Data Cleaning")
        st.markdown("""
        The data cleaning process involved the following steps:

        1. **Outlier Removal:** 
           - **Method:** Interquartile Range (IQR) method.
           - **Details:** Removed values outside the range of `Q1 - 2*IQR` and `Q3 + 2*IQR` for the following numerical columns:
           - `property_age`, `property_size`, `rent`, `deposit`, `photo_count`.
        
        2. **Missing Value Imputation:**
           - **Numerical Columns:** Filled missing values with the mean of each respective column.
           - **Categorical Columns:** Filled missing values with the mode (most frequent value) of each respective column.
        
        3. **Categorical Variable Encoding:** 
           - Applied One-Hot Encoding to categorical variables to convert them into a suitable numerical format for modeling.
        
        4. **Feature Scaling:** 
           - Scaled numerical features using Min-Max Scaling to normalize the data.
        """)

    else:
        st.warning("Please upload all three data files to proceed.")

# Data Visualization Section
elif selection == "Data Visualization":
    st.title("Data Visualization")
    
    # Sidebar Upload Data Files
    with st.sidebar.expander("📤 Upload Data Files"):
        data_file = st.file_uploader("Upload `property_data_set.csv`", type=["csv"])
        interaction_file = st.file_uploader("Upload `property_interactions.csv`", type=["csv"])
        pics_file = st.file_uploader("Upload `property_photos.tsv`", type=["tsv"])

    if data_file and interaction_file and pics_file:
        # Load and process data
        data, interaction, pics, dataset = load_and_process_data(data_file, interaction_file, pics_file)

        # Categorical variable selection for Count Plot
        categorical_vars = dataset.select_dtypes(include=['object']).columns.tolist()

        # 1. Histogram Plot
        st.header("Histogram of Number of Requests")
        with st.container():
            category_choice_hist = st.radio(
                "Select Category for Histogram",
                options=['3 Days', '7 Days'],
                index=0,
                key='hist_category'
            )
            if category_choice_hist == '3 Days':
                hist_data = dataset['request_day_within_3d']
                title = 'Histogram of Number of Requests in First 3 Days'
            else:
                hist_data = dataset['request_day_within_7d']
                title = 'Histogram of Number of Requests in First 7 Days'

            st.altair_chart(
                alt.Chart(pd.DataFrame({'request_days': hist_data})).mark_bar().encode(
                    alt.X('request_days:Q', bin=alt.Bin(maxbins=30)),
                    alt.Y('count()', title='Count'),
                    tooltip=['count()']
                ).properties(
                    title=title,
                    width=700,
                    height=400
                ).interactive(),
                use_container_width=True
            )

        # 2. Value Counts for Categories
        st.header("Value Counts for Categories")
        with st.container():
            category_choice_cat = st.radio(
                "Select Category for Value Counts",
                options=['3 Days', '7 Days'],
                index=0,
                key='cat_category'
            )
            if category_choice_cat == '3 Days':
                cat_data = dataset['categories_3day']
                title = 'Value Count for Each Category Within 3 Days'
            else:
                cat_data = dataset['categories_7day']
                title = 'Value Count for Each Category Within 7 Days'

            fig, ax = plt.subplots(figsize=(10, 6))
            palette = "viridis" if category_choice_cat == '3 Days' else "magma"
            sns.countplot(y=cat_data, ax=ax, palette=palette)
            ax.set_title(title)
            st.pyplot(fig)

        # 3. Pairplot (Using Seaborn as Altair does not support pairplot directly)
        st.header("Pairplot of Selected Features")
        with st.container():
            selected_features = st.multiselect(
                "Select Numeric Features for Pairplot",
                options=['property_age', 'property_size', 'rent', 'deposit', 'photo_count'],
                default=['property_age', 'property_size']
            )
            if selected_features:
                if category_choice_hist == '3 Days':
                    pairplot_data = dataset[selected_features + ['request_day_within_3d']].dropna()
                    pairplot_title = "Pairplot of Selected Features vs. Requests Within 3 Days"
                else:
                    pairplot_data = dataset[selected_features + ['request_day_within_7d']].dropna()
                    pairplot_title = "Pairplot of Selected Features vs. Requests Within 7 Days"

                fig = sns.pairplot(pairplot_data, diag_kind='kde')
                fig.fig.suptitle(pairplot_title, y=1.03)
                st.pyplot(fig)
            else:
                st.warning("Please select at least one feature to display the pairplot.")

        # 4. Interactive Scatter Matrix (Plotly)
        st.header("Interactive Scatter Matrix")
        with st.container():
            scatter_features = st.multiselect(
                "Select Features for Scatter Matrix",
                options=['property_age', 'property_size', 'rent', 'deposit', 'photo_count'],
                default=['property_age', 'property_size']
            )
            if category_choice_hist == '3 Days':
                scatter_data = dataset[scatter_features + ['request_day_within_3d']].dropna()
                scatter_title = "Scatter Matrix for Selected Features (3 Days)"
            else:
                scatter_data = dataset[scatter_features + ['request_day_within_7d']].dropna()
                scatter_title = "Scatter Matrix for Selected Features (7 Days)"

            if len(scatter_features) >= 1:
                fig = px.scatter_matrix(scatter_data,
                                        dimensions=scatter_features + ([ 'request_day_within_3d' ] if category_choice_hist == '3 Days' else ['request_day_within_7d']),
                                        title=scatter_title,
                                        template="simple_white",
                                        height=800, width=800)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one feature to display the scatter matrix.")

        # 5. Correlation Heatmap (Altair)
        st.header("Correlation Heatmap")
        with st.container():
            heatmap_features = st.multiselect(
                "Select Features for Heatmap",
                options=dataset.select_dtypes(include=[np.number]).columns.tolist(),
                default=dataset.select_dtypes(include=[np.number]).columns.tolist()
            )
            if len(heatmap_features) > 1:
                corr = dataset[heatmap_features].corr().reset_index().melt('index')
                corr.columns = ['Feature 1', 'Feature 2', 'Correlation']

                # Choose a valid Altair color scheme
                heatmap_color_scheme = 'bluegreen'  # Replaced 'YlGnBu' with 'bluegreen'

                heatmap = alt.Chart(corr).mark_rect().encode(
                    x=alt.X('Feature 1:O', sort=None),
                    y=alt.Y('Feature 2:O', sort=None),
                    color=alt.Color('Correlation:Q', scale=alt.Scale(scheme=heatmap_color_scheme)),
                    tooltip=['Feature 1', 'Feature 2', 'Correlation']
                ).properties(
                    width=700,
                    height=700,
                    title="Correlation Heatmap"
                )

                text = heatmap.mark_text(baseline='middle').encode(
                    text=alt.Text('Correlation:Q', format=".2f"),
                    color=alt.condition(
                        alt.datum.Correlation > 0.5,
                        alt.value('white'),
                        alt.value('black')
                    )
                )

                combined_heatmap = heatmap + text
                st.altair_chart(combined_heatmap, use_container_width=True)
            else:
                st.warning("Please select at least two features for the heatmap.")

        # 6. Count Plot for Selected Categorical Variable
        st.header("Count Plot for Categorical Variable")
        with st.container():
            if categorical_vars:
                selected_categorical = st.selectbox(
                    "Select Categorical Variable for Count Plot",
                    options=categorical_vars
                )
                if selected_categorical:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(y=dataset[selected_categorical], order=dataset[selected_categorical].value_counts().index, ax=ax, palette="coolwarm")
                    ax.set_title(f'Count Plot for {selected_categorical}')
                    st.pyplot(fig)
            else:
                st.warning("No categorical variables found in the dataset.")

    else:
        st.warning("Please upload all three data files to proceed.")

# Model Training and Evaluation Section
elif selection == "Model Training and Evaluation":
    st.title("Model Training and Evaluation")
    
    # Check if files are uploaded
    with st.sidebar.expander("📤 Upload Data Files"):
        data_file = st.file_uploader("Upload `property_data_set.csv`", type=["csv"])
        interaction_file = st.file_uploader("Upload `property_interactions.csv`", type=["csv"])
        pics_file = st.file_uploader("Upload `property_photos.tsv`", type=["tsv"])

    if data_file and interaction_file and pics_file:
        # Load and process data
        data, interaction, pics, dataset = load_and_process_data(data_file, interaction_file, pics_file)

        # Prepare data for modeling and get cleaning steps
        category_choice = st.radio(
            "Select Category for Modeling",
            options=['3 Days', '7 Days'],
            index=0
        )
        data_with_days, target_variable, cleaning_steps = prepare_modeling_data(dataset, category_choice)

        # Display target variable
        st.write(f"**Target Variable:** {target_variable}")

        # Create a two-column layout: Left for outputs, Right for model options
        col1, col2 = st.columns([3, 1])

        with col2:
            st.markdown("### 🤖 **Model Options**")
            model_type = st.selectbox("Select Model Type", options=['Linear Regression', 'KNN Regressor'])
            if model_type == 'KNN Regressor':
                n_neighbors = st.slider("Select Number of Neighbors (K)", min_value=1, max_value=20, value=5)
            else:
                n_neighbors = None

        with col1:
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

            # Plot Actual vs Predicted
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


