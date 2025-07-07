# Project Setup Instructions

This guide provides instructions to set up and run the project.

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Requirements**:
   Install the required Python packages using the following command:
   ```bash
   pip install groq==0.5.0 sentence-transformers==2.6.1 streamlit==1.33.0 PyPDF2==3.0.1 transformers==4.40.0 shap==0.45.0 lime==0.2.0.1 python-dotenv==1.0.0 xgboost==3.0.2 faiss-cpu==1.7.4
   ```

## Running the Application

1. **Navigate to the Project Directory**:
   Ensure you are in the root directory of the project where `app/main.py` is located.

2. **Run the Streamlit Application**:
   Execute the following command to start the Streamlit app:
   ```bash
   streamlit run app/main.py
   ```

3. **Access the Application**:
   Once the command runs, Streamlit will start a local server and provide a URL (typically `http://localhost:8501`). Open this URL in your web browser to view the application.

## Notes
- Ensure all dependencies are installed correctly before running the application.
- If you encounter any issues, verify that the Python environment is activated and the required packages are installed.
- For environment-specific configurations, create a `.env` file in the project root and set necessary variables (refer to project documentation for details).