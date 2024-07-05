import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from fuzzywuzzy import process
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def read_dataset(file_path):
    return pd.read_csv(file_path)

# ... (keep all the other functions as they are) ...

# Load the Mistral 7B model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def extract_columns(query, data):
    words = query.lower().split()
    possible_columns = [word for word in words if word in data.columns]
    return possible_columns

def process_query(query, data):
    try:
        # Use Mistral 7B to understand the query
        context = ', '.join(data.columns)
        prompt = f"Given the following columns in a dataset: {context}\n\nUser query: {query}\n\nWhat type of analysis should be performed and which columns should be used?"
        result = generate_response(prompt)
        
        # Fuzzy match for test types
        test_types = ["chi-square", "t-test", "correlation", "pearson", "summary", "histogram", "scatter plot"]
        best_match, score = process.extractOne(result, test_types)
        
        if score < 60:  # Adjust this threshold as needed
            return "I'm not sure what analysis you want to perform. Could you please rephrase your query?"

        columns = extract_columns(query, data)

        if best_match in ["chi-square", "t-test", "correlation", "pearson"] and len(columns) >= 2:
            col1, col2 = columns[:2]
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
        elif best_match == "histogram" and len(columns) >= 1:
            column = columns[0]
            fig = plot_histogram(data, column)
            return fig, f"Histogram for {column} plotted."
        elif best_match == "scatter plot" and len(columns) >= 2:
            col1, col2 = columns[:2]
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