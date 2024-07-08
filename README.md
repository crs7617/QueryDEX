# queryDex

queryDex is an interactive data analysis tool that allows users to explore and analyze datasets using natural language queries. It leverages the power of Streamlit for the user interface and Google's Generative AI to interpret user queries and perform appropriate statistical analyses.

## Features

- **Upload CSV or Excel datasets**
- **Perform various statistical analyses:**
  - Chi-square test
  - T-test
  - Pearson correlation
  - Summary statistics
  - And more

- **Visualize data with:**
  - Histograms
  - Scatter plots
  - Box plots
  - Line charts

- **Natural language query processing**
- **Interactive web interface**

## Installation

1. **Clone this repository:**

    ```sh
    git clone https://github.com/yourusername/queryDex.git
    cd queryDex
    ```

2. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Set up your environment variables:**

    Create a `.env` file in the root directory and add your Google Generative AI API key:

    ```sh
    GEMINI_API_KEY=your_api_key_here
    ```

## Usage

1. **Run the Streamlit app:**

    ```sh
    streamlit run app.py
    ```

2. **Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).**

3. **Upload your CSV or Excel dataset using the file uploader.**

4. **Enter your query in natural language. For example:**
    - "Show me a histogram of age"
    - "Is there a correlation between height and weight?"
    - "Perform a t-test on salary between male and female employees"
    - "Plot a box plot of income by region"
    - "Display a line chart of sales over time"

5. **View the results of your query, including statistical test results or visualizations.**

## How It Works

1. The app uses Streamlit for the user interface, allowing for easy data upload and query input.
2. User queries are processed using Google's Generative AI to determine the type of analysis requested.
3. The app extracts relevant column names from the query and matches them to the dataset.
4. Depending on the analysis type, the app performs the appropriate statistical test or generates the requested visualization.
5. Results are displayed directly in the Streamlit interface.




https://github.com/crs7617/QueryDEX/assets/115174268/332a533f-d5fd-4d9e-83f6-f467cfdc7b48



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Streamlit for the web app framework
- Google Generative AI for natural language processing
- pandas for data manipulation
- scipy for statistical computations
- matplotlib and seaborn for data visualization
