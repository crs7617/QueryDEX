import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from fuzzywuzzy import process
import google.generativeai as genai

# Configure the Gemini API
genai.configure(api_key='AIzaSyBirAbeBRI9lUmD5mF8i5ZOGSHDQS9f3fg')

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

def read_dataset(file_path):
    return pd.read_csv(file_path)

def extract_columns(query, data):
    words = query.lower().split()
    possible_columns = [col for col in data.columns if col.lower() in words]
    if len(possible_columns) >= 2:
        return possible_columns[:2]
    elif len(possible_columns) == 1:
        return possible_columns
    return None

def get_actual_column(data, column):
    return next(col for col in data.columns if col.lower() == column.lower())

def chi_square_test(data, col1, col2):
    actual_col1 = get_actual_column(data, col1)
    actual_col2 = get_actual_column(data, col2)
    contingency_table = pd.crosstab(data[actual_col1], data[actual_col2])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    return chi2, p

def t_test(data, col1, col2):
    actual_col1 = get_actual_column(data, col1)
    actual_col2 = get_actual_column(data, col2)
    t_stat, p_val = ttest_ind(data[actual_col1], data[actual_col2])
    return t_stat, p_val

def pearson_corr(data, col1, col2):
    actual_col1 = get_actual_column(data, col1)
    actual_col2 = get_actual_column(data, col2)
    corr, p_val = pearsonr(data[actual_col1], data[actual_col2])
    return corr, p_val

def data_summary(data):
    return data.describe()

def plot_histogram(data, column):
    actual_column = get_actual_column(data, column)
    fig, ax = plt.subplots()
    sns.histplot(data[actual_column], kde=True, ax=ax)
    return fig

def plot_scatter(data, col1, col2):
    actual_col1 = get_actual_column(data, col1)
    actual_col2 = get_actual_column(data, col2)
    fig, ax = plt.subplots()
    sns.scatterplot(x=data[actual_col1], y=data[actual_col2], ax=ax)
    return fig

def process_query(query, data):
    try:
        prompt = f"Given the following columns in a dataset: {', '.join(data.columns)}, " \
                 f"what type of analysis is being requested in this query: '{query}'? " \
                 f"Possible analysis types are: chi-square test, t-test, correlation, pearson correlation, " \
                 f"summary statistics, histogram, or scatter plot. Only return the analysis type."
        
        response = model.generate_content(prompt)
        result = response.text.strip().lower()

        test_types = ["chi-square", "t-test", "correlation", "pearson", "summary", "histogram", "scatter plot"]
        best_match, score = process.extractOne(result, test_types)
        
        if score < 60:
            return "I'm not sure what analysis you want to perform. Could you please rephrase your query?"

        columns = extract_columns(query, data)

        if best_match in ["chi-square", "t-test", "correlation", "pearson"] and len(columns) == 2:
            col1, col2 = columns
            if best_match == "chi-square":
                chi2, p = chi_square_test(data, col1, col2)
                return f"Chi-square test result between {col1} and {col2}: chi2={chi2}, p={p}"
            elif best_match == "t-test":
                t_stat, p_val = t_test(data, col1, col2)
                return f"T-test result between {col1} and {col2}: t_stat={t_stat}, p_val={p_val}"
            elif best_match in ["correlation", "pearson"]:
                corr, p_val = pearson_corr(data, col1, col2)
                return f"Pearson correlation result between {col1} and {col2}: corr={corr}, p_val={p_val}"
        elif best_match == "summary":
            return data_summary(data)
        elif best_match == "histogram" and columns:
            column = columns[0]
            fig = plot_histogram(data, column)
            return fig, f"Histogram for {column} plotted."
        elif best_match == "scatter plot" and len(columns) == 2:
            col1, col2 = columns
            fig = plot_scatter(data, col1, col2)
            return fig, f"Scatter plot for {col1} and {col2} plotted."
        else:
            return "I couldn't understand the columns to analyze. Please specify the column names clearly."

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit App
st.title('queryDex')

st.header('Upload Dataset')
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    data = read_dataset(uploaded_file)
    st.write("Dataset Uploaded Successfully")
    st.write(data.head())

    st.header('Query Dataset')
    user_query = st.text_input("Enter your query")
    
    if user_query:
        result = process_query(user_query, data)
        if isinstance(result, tuple):
            fig, message = result
            st.pyplot(fig)
            st.write(message)
        elif isinstance(result, pd.DataFrame):
            st.write(result)
        else:
            st.write(result)