# AI Resume Match Maker

This project is designed to match resumes with job descriptions using AI and machine learning techniques. The project involves setting up a Qdrant vector database, uploading data, and running a Streamlit application to display the results.

## Requirements

Make sure you have the following installed on your system:

- Python 3.x
- Docker
- Streamlit

## Installation

1. **Clone the repository**:
    ```sh
    download the zipfile
    ```

2. **Install Python dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run Qdrant**:
    ```sh
    docker run -p 6333:6333 -v .:/qdrant/storage qdrant/qdrant
    ```

4. **Upload data to Qdrant**:
    ```sh
    py upload.py
    ```

5. **Run the Streamlit application**:
    ```sh
    streamlit run Demo.py
    ```


## Usage

After running the commands above, open your web browser and navigate to `http://localhost:8501` to view the Streamlit application. The app allows you to upload resumes and job descriptions, process them, and view the matching results.

## Troubleshooting

If you encounter any issues, make sure that:
- Docker is running correctly on your system.
- All dependencies in `requirements.txt` are installed.
- Ports are not blocked by other applications.

For further assistance, refer to the project's documentation or contact the project maintainers.

## Acknowledgements

- Qdrant for the vector database.
- Streamlit for the web application framework.
- Contributors and maintainers of this project.

