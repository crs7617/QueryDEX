import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import streamlit as st
from fuzzywuzzy import process
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


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
    fig, ax = plt.subplots()
    sns.histplot(data[column], ax=ax)
    ax.set_title(f'Histogram of {column}')
    return fig

def plot_scatter(data, col1, col2):
    fig, ax = plt.subplots()
    sns.scatterplot(x=data[col1], y=data[col2], ax=ax)
    ax.set_title(f'Scatter plot of {col1} vs {col2}')
    return fig

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad" 
nlp_model = pipeline("question-answering", model=model_name, tokenizer=model_name)

def extract_columns(query, data):
    words = query.lower().split()
    possible_columns = [word for word in words if word in data.columns]
    if len(possible_columns) >= 2:
        return possible_columns[:2]
    return None

def process_query(query, data):
    try:
        # Use NLP model to understand the query
        result = nlp_model(question=query, context=', '.join(data.columns))
        
        # Fuzzy match for test types
        test_types = ["chi-square", "t-test", "correlation", "pearson", "summary", "histogram", "scatter plot"]
        best_match, score = process.extractOne(result['answer'], test_types)
        
        if score < 60:  # Adjust this threshold as needed
            return "I'm not sure what analysis you want to perform. Could you please rephrase your query?"

        columns = extract_columns(query, data)

        if best_match in ["chi-square", "t-test", "correlation", "pearson"] and columns:
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
            return data_summary(data).to_dict()
        elif best_match == "histogram" and columns:
            column = columns[0]
            fig = plot_histogram(data, column)
            return fig, f"Histogram for {column} plotted."
        elif best_match == "scatter plot" and columns:
            col1, col2 = columns
            fig = plot_scatter(data, col1, col2)
            return fig, f"Scatter plot for {col1} and {col2} plotted."
        else:
            return "I couldn't understand the columns to analyze. Please specify the column names clearly."

    except Exception as e:
        return f"An error occurred: {str(e)}"

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
        if isinstance(result, tuple):
            fig, message = result
            st.pyplot(fig)
            st.write(message)
        else:
            st.write(result)

        if isinstance(result, dict):
            st.write(pd.DataFrame(result))