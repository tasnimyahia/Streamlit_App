import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import StringIO

def describe_data(df):
    """Generates descriptive statistics for the DataFrame."""
    return df.describe()

def visualize_data(df, column):
    """Generates visualizations for the selected column."""
    fig, ax = plt.subplots()
    sns.histplot(df[column], ax=ax, kde=True)
    plt.title(f"Histogram of {column} with KDE")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df[column], ax=ax2)
    plt.title(f"Box Plot of {column}")
    return fig, fig2

def correlation_matrix(df):
    """Generates a correlation matrix for the DataFrame."""
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])
    if numeric_cols.empty:
        st.warning("No numeric columns found for correlation analysis.")
        return None
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_cols.corr(), annot=True, ax=ax, cmap='coolwarm')
    plt.title("Correlation Matrix (Numeric Columns)")
    return fig

def missing_value_analysis(df):
    """Displays the number of missing values per column."""
    missing_values = df.isnull().sum()
    st.write("Missing Values per Column:")
    st.table(missing_values)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=True, ax=ax)
    plt.title("Missing Values Heatmap")
    st.pyplot(fig)

def handle_missing_values(df, method="mean", column=None):
    """Handles missing values based on the selected method and column."""
    df_copy = df.copy()  # Work on a copy of the dataframe to preview changes

    if method == "mean":
        if column:
            df_copy[column].fillna(df_copy[column].mean(), inplace=True)
        else:
            df_copy.fillna(df_copy.mean(), inplace=True)
    elif method == "median":
        if column:
            df_copy[column].fillna(df_copy[column].median(), inplace=True)
        else:
            df_copy.fillna(df_copy.median(), inplace=True)
    elif method == "mode":
        if column:
            df_copy[column].fillna(df_copy[column].mode()[0], inplace=True)
        else:
            df_copy.fillna(df_copy.mode().iloc[0], inplace=True)
    elif method == "drop":
        if column:
            df_copy.dropna(subset=[column], inplace=True)
        else:
            df_copy.dropna(inplace=True)
    else:
        st.error("Invalid method for handling missing values.")
    
    return df_copy

def handle_duplicates(df):
    """Handles duplicate rows in the DataFrame."""
    num_duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {num_duplicates}")
    
    # Preview the effect of removing duplicates
    if num_duplicates > 0:
        df_preview = df.copy()  # Make a copy to preview the effect
        df_preview.drop_duplicates(inplace=True)

        # Preview the data before and after removing duplicates
        st.subheader("Data Before Removing Duplicates")
        st.write("Number of duplicate rows:", df.duplicated().sum())
        
        st.subheader("Data After Removing Duplicates")
        st.write("Number of duplicate rows:" ,df_preview.duplicated().sum())

    else:
        st.write("No duplicate rows found.")

    return df

def outlier_analysis(df, column):
    """Identifies and displays outliers using the IQR method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    st.write(f"Number of outliers in {column}: {len(outliers)}")
    if not outliers.empty:
        st.write(outliers)
        show_outliers_vis = st.checkbox("Show outliers visualization", key='show_outliers_vis')
        if show_outliers_vis:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[column], ax=ax)
            sns.scatterplot(x=outliers[column], y=[0]*len(outliers), color='red', marker='o', ax=ax)
            plt.title(f"Box Plot of {column} with Outliers highlighted")
            st.pyplot(fig)
    return lower_bound, upper_bound

def handle_outliers(df, column, lower_bound, upper_bound, method='clip'):
    """Handles outliers based on the selected method."""
    df_copy = df.copy()  # Make a copy for previewing changes
    
    if method == 'clip':
        df_copy[column] = df_copy[column].clip(lower=lower_bound, upper=upper_bound)
        st.success(f"Outliers in {column} have been clipped to the defined bounds.")
    elif method == 'drop':
        df_copy.drop(df_copy[(df_copy[column] < lower_bound) | (df_copy[column] > upper_bound)].index, inplace=True)
        st.success(f"Outliers in {column} have been removed.")
    else:
        st.error("Invalid method for handling outliers.")
    
    return df_copy

def data_types_analysis(df):
    """Displays data type information and allows conversion."""
    st.header("Data Types Analysis")
    st.write(df.dtypes)
    return df
def column_names_analysis(df):
    st.header("Column Name Analysis")
    st.write("Current Column Names:")
    st.write(df.columns)


    st.subheader("Rename Columns:")
    new_column_names = {}
    for col in df.columns:
        new_name = st.text_input(f"Rename '{col}' to:", value=col, key=f"rename_{col}")
        new_column_names[col] = new_name

    # Preview the DataFrame with new column names
    st.subheader("Preview of Renamed Columns:")
    df_preview = df.rename(columns=new_column_names)
    st.write(df_preview.columns)

    if st.button("Apply Column Renaming"):
        try:
            df.rename(columns=new_column_names, inplace=True)
            st.session_state['data'] = df  # Save the updated DataFrame
            st.success("Columns renamed successfully!")
        except Exception as e:
            st.error(f"Error renaming columns: {e}")
    return df


def download_dataset(df):
    """Downloads the DataFrame as a CSV file."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="downloaded_data.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

def reset_all_flags():
    """Resets all conditional display flags."""
    keys_to_reset = [
        'show_data', 'describe_data', 'missing_analysis_run',
        'missing_values_handled', 'duplicates_handled',
        'outlier_analysis_run', 'outliers_handled',
        'visualize_data_run', 'correlation_run','type_converted','columns_renamed','data_type_analysis_clicked'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            st.session_state[key] = False

