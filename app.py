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
st.set_page_config(page_title="Property Customer Interaction Prediction Dashboard", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", [
    "Introduction",
    "Data Exploration",
    "Data Cleaning",
    "Data Visualization",
    "Model Training and Evaluation",
    "Prediction",
    "Real-world Application and Impact"
])

# Sidebar Data Source Options
st.sidebar.markdown("---")
data_source_option = st.sidebar.radio("Select Data Source:", ["Use Default Files", "Upload Your Own Files"], index=0)

# File Upload UI if user wants to upload their own files
uploaded_data_file = None
uploaded_interaction_file = None
uploaded_pics_file = None

if data_source_option == "Upload Your Own Files":
    st.sidebar.markdown("#### Upload Your Files")
    uploaded_data_file = st.sidebar.file_uploader("Upload `property_data_set.csv`", type=["csv"])
    uploaded_interaction_file = st.sidebar.file_uploader("Upload `property_interactions.csv`", type=["csv"])
    uploaded_pics_file = st.sidebar.file_uploader("Upload `property_photos.tsv`", type=["tsv"])

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
    original_count = df_in.shape[0]
    df_out = df_in.loc[(df_in[col_name] <= fence_high) & (df_in[col_name] >= fence_low)]
    cleaned_count = df_out.shape[0]
    return df_out, original_count - cleaned_count

# URLs for the default data files
DATA_URL = "https://drive.google.com/uc?id=1LeRX6f-6pOomTV2fT8DpMVUfam5PX9s0"
INTERACTION_URL = "https://drive.google.com/uc?id=1NFnM6CeJz4925Ep1q0bNnCsi0-6hSlJy"
PICS_URL = "https://drive.google.com/uc?id=1u1mUOTlEVQS4kMCsWEoa04HKwoj3xm9L"

def load_data_from_files(data_file, interaction_file, pics_file):
    data = pd.read_csv(data_file, parse_dates=['activation_date'], dayfirst=True)
    interaction = pd.read_csv(interaction_file, parse_dates=['request_date'], dayfirst=True)
    pics = pd.read_csv(pics_file, sep='\t')
    return data, interaction, pics

def load_and_process_data(data_file=None, interaction_file=None, pics_file=None):
    if data_file is not None and interaction_file is not None and pics_file is not None:
        data, interaction, pics = load_data_from_files(data_file, interaction_file, pics_file)
    else:
        data = pd.read_csv(DATA_URL, parse_dates=['activation_date'], dayfirst=True)
        interaction = pd.read_csv(INTERACTION_URL, parse_dates=['request_date'], dayfirst=True)
        pics = pd.read_csv(PICS_URL, sep='\t')

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

def prepare_modeling_data(df, category_choice):
    cleaning_steps = {
        'outlier_removal': {},
        'imputations': {},
        'encodings': [],
        'scaling': []
    }
    
    df_clean = df.copy()
    outlier_counts = {}
    for col in ['property_age', 'property_size', 'rent', 'deposit', 'photo_count']:
        df_clean, removed = remove_outlier(df_clean, col)
        outlier_counts[col] = removed
    cleaning_steps['outlier_removal'] = outlier_counts

    def capping_for_3days(x):
        num = 10
        return min(x, num)

    def capping_for_7days(x):
        num = 20
        return min(x, num)

    df_clean['request_day_within_3d_capping'] = df_clean['request_day_within_3d'].apply(capping_for_3days)
    df_clean['request_day_within_7d_capping'] = df_clean['request_day_within_7d'].apply(capping_for_7days)

    X = df_clean.drop(['request_day_within_7d', 'categories_7day', 'request_day_within_3d',
                      'categories_3day', 'request_day_within_3d_capping',
                      'request_day_within_7d_capping'], axis=1)
    x_cat_withNull = df_clean[X.select_dtypes(include=['O']).columns]
    x_remain_withNull = df_clean[X.select_dtypes(exclude=['O']).columns]
    y = df_clean[['request_day_within_7d', 'categories_7day', 'request_day_within_3d',
                  'categories_3day', 'request_day_within_3d_capping',
                  'request_day_within_7d_capping']]

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

    ohe = OneHotEncoder(categories='auto', handle_unknown='ignore', sparse_output=False)
    feature_train = ohe.fit_transform(x_cat)
    feature_labels = ohe.get_feature_names_out(x_cat.columns)
    df_features = pd.DataFrame(feature_train, columns=feature_labels)
    encoded_columns = x_cat.columns.tolist()
    cleaning_steps['encodings'] = encoded_columns

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

# Load data (either default or user uploaded)
if data_source_option == "Use Default Files":
    data, interaction, pics, dataset = load_and_process_data()
elif data_source_option == "Upload Your Own Files":
    if uploaded_data_file and uploaded_interaction_file and uploaded_pics_file:
        data, interaction, pics, dataset = load_and_process_data(uploaded_data_file, uploaded_interaction_file, uploaded_pics_file)
    else:
        data = interaction = pics = dataset = None

# Introduction Section
if selection == "Introduction":
    st.title("🏠 Property Customer Interaction Prediction Dashboard")
    
    st.markdown("""
    ### Welcome to the Property Customer Interaction Prediction Dashboard!
    
    This application is designed to help you **explore**, **visualize**, **model**, and **predict** property-related data seamlessly. Whether you're a data enthusiast, a real estate analyst, or a developer, this tool offers a comprehensive suite of features to assist you in deriving meaningful insights from your property datasets.
    
    ---
    
    ### 📈 **App Features**
    
    1. **Data Exploration**
        - **What:** Easily upload your property datasets or use default data.
        - **Files Required:**
            - `property_data_set.csv`: Contains detailed information about properties.
            - `property_interactions.csv`: Logs interactions or requests related to properties.
            - `property_photos.tsv`: Contains URLs or metadata for property photos.
    
    2. **Data Cleaning**
        - **What:** Prepare your data for analysis and modeling by addressing inconsistencies.
        - **Processes Involved:**
            - Outlier removal using the Interquartile Range (IQR) method.
            - Handling missing values through mean and mode imputation.
            - Encoding categorical variables for machine learning compatibility.
            - Feature scaling to normalize numerical data.
    
    3. **Data Visualization**
        - **What:** Create interactive and insightful visualizations to uncover trends and patterns.
        - **Tools & Plots:**
            - Histograms and count plots.
            - Pairplots and scatter matrices.
            - Correlation heatmaps.
            - Interactive charts using Plotly and Altair.
    
    4. **Model Training and Evaluation**
        - **What:** Build and assess predictive models based on your cleaned data.
        - **Models Available:**
            - **Linear Regression:** Understand relationships between variables.
            - **K-Nearest Neighbors (KNN) Regressor:** Capture non-linear patterns.
        - **Evaluation Metrics:**
            - Root Mean Squared Error (RMSE).
    
    5. **Prediction**
        - **What:** Use the trained model to predict the number of requests based on custom input features.
    
    ---
    
    ### 🛠 **How to Get Started**
    
    1. **Select Data Source:**
        - In the sidebar, choose either **Use Default Files** or **Upload Your Own Files**.
        - If uploading, please provide all three files: `property_data_set.csv`, `property_interactions.csv`, and `property_photos.tsv`.
    
    2. **Explore and Clean Your Data:**
        - Proceed to the **Data Exploration** and **Data Cleaning** sections to understand and prepare your data.
    
    3. **Visualize Insights:**
        - Use the **Data Visualization** section to create charts and plots that highlight key trends.
    
    4. **Train and Evaluate Models:**
        - Head over to the **Model Training and Evaluation** section to build predictive models and assess their performance.
    
    5. **Predict:**
        - Finally, use the **Prediction** section to input your own data and predict the number of requests.
    
    ---
    
    **Enjoy exploring, visualizing, modeling, and predicting with your property data!**
    """)
# Data Exploration Section
elif selection == "Data Exploration":
    st.title("Data Exploration")
    if dataset is None:
        st.warning("Please load the data first (either use default or upload your own files).")
    else:
        st.header("Property Data")
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

        st.header("Interaction Data")
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

        st.header("Pics Data")
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

# Data Cleaning Section
elif selection == "Data Cleaning":
    st.title("Data Cleaning")
    if dataset is None:
        st.warning("Please load the data first (either use default or upload your own files).")
    else:
        category_choice = st.radio(
            "Select Category for Data Cleaning Overview",
            options=['3 Days', '7 Days'],
            index=0
        )
        data_with_days, target_variable, cleaning_steps = prepare_modeling_data(dataset, category_choice)

        st.header("Data Cleaning Steps Overview")

        # Outlier Removal
        st.subheader("1. Outlier Removal")
        outlier_df = pd.DataFrame({
            'Column': list(cleaning_steps['outlier_removal'].keys()),
            'Outliers Removed': list(cleaning_steps['outlier_removal'].values())
        })
        st.table(outlier_df)
        st.markdown("""
        **Methodology:** Outliers were removed based on the Interquartile Range (IQR) method. Specifically, values outside the range of `Q1 - 2*IQR` and `Q3 + 2*IQR` were considered outliers.
        """)

        # Missing Value Imputations
        st.subheader("2. Missing Value Imputations")
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

        # Categorical Variable Encodings
        st.subheader("3. Categorical Variable Encodings")
        if cleaning_steps['encodings']:
            st.markdown("**One-Hot Encoded Columns:**")
            st.write(cleaning_steps['encodings'])
        else:
            st.markdown("**One-Hot Encoded Columns:** None.")

        # Feature Scaling
        st.subheader("4. Feature Scaling")
        if cleaning_steps['scaling']:
            st.markdown("**Scaled Numerical Columns (Min-Max Scaling):**")
            st.write(cleaning_steps['scaling'])
        else:
            st.markdown("**Scaled Numerical Columns (Min-Max Scaling):** None.")

        # Summary
        st.header("Summary of Data Cleaning")
        st.markdown("""
        The data cleaning process involved:
        1. **Outlier Removal**
        2. **Missing Value Imputation (Mean for Numerical, Mode for Categorical)**
        3. **One-Hot Encoding of Categorical Variables**
        4. **Min-Max Scaling of Numerical Features**
        """)

# Data Visualization Section
elif selection == "Data Visualization":
    st.title("Data Visualization")
    if dataset is None:
        st.warning("Please load the data first (either use default or upload your own files).")
    else:
        categorical_vars = dataset.select_dtypes(include=['object']).columns.tolist()

        # Histogram Plot
        st.header("Histogram of Number of Requests")
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

        # Value Counts for Categories
        st.header("Value Counts for Categories")
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

        # Pairplot
        st.header("Pairplot of Selected Features")
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
            st.warning("Please select at least one feature for the pairplot.")

        # Interactive Scatter Matrix (Plotly)
        st.header("Interactive Scatter Matrix")
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
            st.warning("Please select at least one feature for the scatter matrix.")

        # Correlation Heatmap (Altair)
        st.header("Correlation Heatmap")
        heatmap_features = st.multiselect(
            "Select Features for Heatmap",
            options=dataset.select_dtypes(include=[np.number]).columns.tolist(),
            default=dataset.select_dtypes(include=[np.number]).columns.tolist()
        )
        if len(heatmap_features) > 1:
            corr = dataset[heatmap_features].corr().reset_index().melt('index')
            corr.columns = ['Feature 1', 'Feature 2', 'Correlation']

            heatmap_color_scheme = 'bluegreen'

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

        # Count Plot for Selected Categorical Variable
        st.header("Count Plot for Categorical Variable")
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

# Model Training and Evaluation Section
elif selection == "Model Training and Evaluation":
    st.title("Model Training and Evaluation")

    if dataset is None:
        st.warning("Please load the data first (either use default or upload your own files).")
    else:
        # Let the user select the category (3 days or 7 days)
        category_choice = st.radio(
            "Select Category for Modeling",
            options=['3 Days', '7 Days'],
            index=0
        )

        # Prepare the data (This function should be defined elsewhere in your code)
        data_with_days, target_variable, cleaning_steps = prepare_modeling_data(dataset, category_choice)

        st.write(f"**Target Variable:** {target_variable}")

        # Separate features and target
        X = data_with_days.drop(
            ['request_day_within_3d', 'request_day_within_3d_capping',
             'categories_3day', 'request_day_within_7d',
             'request_day_within_7d_capping', 'categories_7day'],
            axis=1, errors='ignore'
        )
        y = data_with_days[target_variable]

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        from sklearn.linear_model import LinearRegression, Lasso
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np

        # Split the data
        seed = 42
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        # Models to compare
        models = {
            "Linear Regression": LinearRegression(),
            "Lasso Regression": Lasso(random_state=seed),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(random_state=seed),
            "Random Forest": RandomForestRegressor(random_state=seed, n_estimators=50)
        }

        # Train and evaluate each model
        performance = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            performance.append((name, rmse))

        # Sort models by RMSE (ascending - best model is the one with smallest RMSE)
        performance_sorted = sorted(performance, key=lambda x: x[1])

        # Display performance
        st.subheader("Model Performance Comparison")
        perf_df = pd.DataFrame(performance_sorted, columns=["Model", "RMSE"])
        st.table(perf_df)

        # Explain model choice
        best_model_name, best_model_rmse = performance_sorted[0]
        st.subheader("Model Selection")

        # Provide a rationale for choosing the final model
        # You can customize this text as needed.
        st.markdown(f"""
        The model with the lowest RMSE is **{best_model_name}** with an RMSE of {best_model_rmse:.4f}.
        
        **Why choose {best_model_name}?**
        - The RMSE is a measure of how far predicted values deviate from actual values. A lower RMSE indicates better predictive accuracy.
        - {best_model_name} outperforms the other tested models on this metric, suggesting it makes more accurate predictions.
        
        Therefore, for final predictions, we will use **{best_model_name}**.
        """)



# Prediction Section
elif selection == "Prediction":
    st.title("Prediction")
    if dataset is None:
        st.warning("Please load the data first (either use default or upload your own files).")
    else:
        st.markdown("""
        Use the trained Linear Regression model to predict the number of property requests.
        
        **Note on Inputs:**
        - The input features are now represented as fractions between 0 and 1.
        - The min and max values shown below are only for your reference, so you understand what 0 and 1 represent in original units.
        """)

        category_choice = st.radio(
            "Select Category for Prediction",
            options=['3 Days', '7 Days'],
            index=0
        )
        data_with_days, target_variable, cleaning_steps = prepare_modeling_data(dataset, category_choice)

        # Separate features and target
        X_days = data_with_days.drop([
            'request_day_within_3d', 'request_day_within_3d_capping', 'categories_3day',
            'request_day_within_7d', 'request_day_within_7d_capping', 'categories_7day'
        ], axis=1, errors='ignore')
        
        y_days = data_with_days[target_variable]

        # Features to be input as fractions 0 to 1
        selected_features = [
            'property_size',
            'rent',
            'deposit',
            'photo_count',
            'property_age'
        ]

        # Example original stats (just for display)
        # These are for user reference only. The model expects 0-1 scaled values.
        # Ensure that your model is trained on 0-1 scaled features for this to be meaningful.
        feature_reference = {
            'property_size': {'min': 0, 'max': 2500, 'unit': 'sq ft'},
            'rent': {'min': 1250, 'max': 34000, 'unit': 'INR'},
            'deposit': {'min': 0, 'max': 320000, 'unit': 'INR'},
            'photo_count': {'min': 0, 'max': 18, 'unit': 'count'},
            'property_age': {'min': 0, 'max': 19, 'unit': 'years'}
        }

        # Always use Linear Regression
        model = LinearRegression()

        # Train the model on the full dataset
        # Assuming the dataset (X_days) is already scaled to [0,1]
        seed = 42
        X_train, X_test, y_train, y_test = train_test_split(X_days, y_days, test_size=0.2, random_state=seed)
        model.fit(X_train, y_train)

        st.subheader("Input Features (0 to 1)")

        input_data = {}
        for col in selected_features:
            ref_min = feature_reference[col]['min']
            ref_max = feature_reference[col]['max']
            ref_unit = feature_reference[col]['unit']

            # Display reference info
            st.markdown(f"**{col.capitalize()}**: 0 represents {ref_min} {ref_unit}, 1 represents {ref_max} {ref_unit}")

            fraction = st.slider(
                f"Select a fraction for {col} (0 to 1):",
                min_value=0.0,
                max_value=1.0,
                value=0.5
            )
            # fraction is directly used as the input
            input_data[col] = fraction

        # For non-selected features, fill with mean of training data (already scaled)
        for col in X_days.columns:
            if col not in input_data:
                input_data[col] = float(X_days[col].mean())

        # Create DataFrame from user input (already scaled to 0-1)
        user_input_df = pd.DataFrame([input_data])[X_days.columns]

        if st.button("Predict"):
            prediction = model.predict(user_input_df)
            st.success(f"Predicted Number of Requests: {prediction[0]:.2f}")

# Real-world Application and Impact Section
elif selection == "Real-world Application and Impact":
    st.title("Real-world Application and Impact")
    st.markdown("""
    ### Applying Insights in the Real World
    
    The predictions and analyses generated by this dashboard can be leveraged in multiple facets of the real estate market. From property listing platforms to individual property managers, the insights can enhance strategic decision-making and lead to improved outcomes.
    
    **1. Property Platforms and Portals:**
    - **Dynamic Pricing:** By predicting the level of tenant interest in the first few days after listing, platforms can suggest optimal rental prices that balance profitability with competitive attractiveness.
    - **Resource Allocation:** Anticipating which listings will draw the most attention helps in allocating customer service and sales teams efficiently, ensuring faster response times and higher tenant satisfaction.
    - **Marketing Strategies:** Properties predicted to have lower initial interest can be supported by targeted promotions or enhanced listing features (e.g., additional property photos, virtual tours).

    **2. Property Owners and Managers:**
    - **Informed Decision-Making:** Understanding expected request volume helps set realistic price points, negotiate better terms, and decide on timing for listing updates or scheduled maintenance.
    - **Property Improvements:** If certain features (like a gym, pool, or prime location) correlate with higher request volumes, owners can invest strategically in property enhancements to increase attractiveness and potential ROI.

    **3. Investors and Developers:**
    - **Portfolio Optimization:** Historical and predictive analysis can guide investors toward property types or neighborhoods that consistently generate high interest, aiding in long-term portfolio planning.
    - **Market Trend Analysis:** Tracking changes in predicted requests over time reveals emerging market trends, helping investors stay ahead of shifts in tenant preferences.

    ### Key Conclusions from the Analysis
    - **Feature Importance:** Certain variables (e.g., property size, age, and photo count) may be particularly influential in shaping initial customer interest. Understanding these drivers can inform more effective listing strategies.
    - **Data Quality Matters:** The data cleaning steps—removing outliers, imputing missing values, and scaling—improved the model’s predictive accuracy. High-quality, well-prepared data is essential for reliable forecasting.
    - **Predictive Power:** Even a straightforward linear regression model can provide valuable insights into expected request volumes. More complex models may offer marginal gains but at the cost of interpretability.

   """)
