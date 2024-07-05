import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import streamlit as st


def read_dataset(file_path):
    return pd.read_csv(file_path)

def chi_square_test(data, col1, col2):
    contingency_table = pd.crosstab(data[col1], data[col2])
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    return chi2, p

def t_test(data, col1, col2):
    t_stat, p_val = ttest_ind(data[col1], data[col2], nan_policy='omit')
    return t_stat, p_val

def pearson_corr(data, col1, col2):
    corr, p_val = pearsonr(data[col1], data[col2])
    return corr, p_val

def data_summary(data):
    return data.describe()

def plot_histogram(data, column):
    sns.histplot(data[column])
    plt.title(f'Histogram of {column}')
    plt.show()

def plot_scatter(data, col1, col2):
    sns.scatterplot(x=data[col1], y=data[col2])
    plt.title(f'Scatter plot of {col1} vs {col2}')
    plt.show()

nlp_model = pipeline("question-answering", model="distilbert-base-uncased", tokenizer="distilbert-base-uncased")

def process_query(query, data):
    query = query.lower()

    if "chi-square" in query:
        columns = query.split("between")[-1].strip().split("and")
        chi2, p = chi_square_test(data, columns[0].strip(), columns[1].strip())
        return f"Chi-square test result: chi2={chi2}, p={p}"
    
    elif "t-test" in query:
        columns = query.split("between")[-1].strip().split("and")
        t_stat, p_val = t_test(data, columns[0].strip(), columns[1].strip())
        return f"T-test result: t_stat={t_stat}, p_val={p_val}"
    
    elif "correlation" in query or "pearson" in query:
        columns = query.split("between")[-1].strip().split("and")
        corr, p_val = pearson_corr(data, columns[0].strip(), columns[1].strip())
        return f"Pearson correlation result: corr={corr}, p_val={p_val}"
    
    elif "summary" in query:
        return data_summary(data).to_dict()
    
    elif "histogram" in query:
        column = query.split("of")[-1].strip()
        plot_histogram(data, column)
        return f"Histogram for {column} plotted."
    
    elif "scatter plot" in query:
        columns = query.split("between")[-1].strip().split("and")
        plot_scatter(data, columns[0].strip(), columns[1].strip())
        return f"Scatter plot for {columns[0]} and {columns[1]} plotted."
    
    else:
        return "Query not recognized. Please try again with a different query."

# Streamlit App
st.title('DataWhiz')

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
        st.write(result)

        if "summary" in user_query:
            st.write(data_summary(data))
        
        if "histogram" in user_query:
            column = user_query.split("of")[-1].strip()
            st.pyplot(plot_histogram(data, column))
        
        if "scatter plot" in user_query:
            columns = user_query.split("between")[-1].strip().split("and")
            st.pyplot(plot_scatter(data, columns[0].strip(), columns[1].strip()))
