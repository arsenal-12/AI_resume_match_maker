import streamlit as st
import fitz
import warnings
from qdrant_client import QdrantClient
import re
import os
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import textwrap
from openai import OpenAI
from qdrant_client.http import models
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config import OPEN_API_KEY

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize Qdrant Client
qdrant_client = QdrantClient("http://localhost:6333")

# Initialize Spacy model
nlp = spacy.load('en_core_web_sm')

# Initialize OpenAI
client = OpenAI(api_key=OPEN_API_KEY)

# Initialize embedding model
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    model_name = "BAAI/bge-large-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

# Function to read PDF
def read_pdf(file_path):
    pdf_text = ""
    document = fitz.open(file_path)
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        pdf_text += page.get_text()
    return pdf_text

# Function to clean text
def clean_text(text):
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[\\/]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to normalize text
def normalize_text(text, use_stemming=False):
    text = text.lower()
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
    if use_stemming:
        tokens = [PorterStemmer().stem(word) for word in tokens]
    else:
        doc = nlp(' '.join(tokens))
        tokens = [token.lemma_ for token in doc]
    temp = ' '.join(tokens)
    formatted_data = "{'skills':'c,java,python,etc..','experience':'commaseperatedin string format','education':'education in string format'}"
    system_message = "You are helping me in an 'AI Resume Match maker' project"
    prompt = f"Parse the skills, education, experience into a Python dictionary for this data {temp}. Do not disturb the internal data. Convert it into a string format 'in triple quotes for avoiding errors' such as {formatted_data}. Ensure the format is ready to use with the eval() method and make it a single line. Ensure give a correct dictionary as output."
    result = get_result(system_message, prompt)
    return result

# Function to get result from OpenAI
def get_result(system, prompt):
    completion = client.chat_completions.create(
        model="gpt-3.5-turbo",
        temperature=0.2,
        n=1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# Function to retrieve point by ID from Qdrant
def retrieve_point_by_id(collection_name, point_id):
    try:
        result = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[point_id]
        )
        if result:
            for point in result:
                text = point.payload.get('text')
                return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Function to insert data into Qdrant
def insert(collection_name, texts):
    embeddings = embeddings_model.embed_documents(texts)
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=len(embeddings[0]), distance=models.Distance.COSINE)
    )
    points = [
        models.PointStruct(id=i+1, vector=embedding, payload={"text": texts[i]})
        for i, embedding in enumerate(embeddings)
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)

# Streamlit App UI
st.set_page_config(page_title="AI Resume Match Maker", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #f4f4f9;
    }
    .main {
        background-color: #ffffff;
        color: #333;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        cursor: pointer;
        font-size: 16px;
        margin: 10px 0;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput, .stTextArea, .stFileUploader>label {
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #ccc;
        margin-bottom: 20px;
        font-size: 16px;
    }
    .stFileUploader>label {
        color: #4CAF50;
        font-weight: bold;
    }
    .stTextArea {
        background-color: #f4f4f9;
    }
    .stTextInput, .stTextArea {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

def homepage():
    st.title("🌟 AI Resume Match Maker 🌟")
    st.write("Welcome to the AI Resume Match Maker. Upload your resume and job description to get started.")
    st.image("https://via.placeholder.com/800x200.png?text=AI+Resume+Match+Maker", use_column_width=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Upload Resume"):
            st.session_state.page = 'upload_resume'
    with col2:
        if st.button("Input Job Description"):
            st.session_state.page = 'input_jd'

def upload_resume():
    st.title("📄 Upload Your Resume")
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx'], label_visibility='collapsed')
    if uploaded_file:
        st.session_state.resume = uploaded_file
        st.session_state.page = 'input_jd'
    st.info("Supported formats: PDF, DOCX")

def input_jd():
    st.title("📝 Input Job Description")
    job_description = st.text_area("Paste the job description here", height=300, label_visibility='collapsed')
    if st.button("Match"):
        st.session_state.job_description = job_description
        st.session_state.page = 'results'
    st.info("Or you can upload a job description file:")
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx'], label_visibility='collapsed')
    if uploaded_file:
        st.session_state.job_description = extract_text_from_file(uploaded_file)
        st.session_state.page = 'results'

def results():
    st.title("🔍 Match Results")
    resume = st.session_state.get('resume')
    job_description = st.session_state.get('job_description')
    if resume and job_description:
        resume_text = read_pdf(resume)
        normalized_resume_text = normalize_text(resume_text)
        normalized_job_description_text = normalize_text(job_description)
        texts = [normalized_resume_text]
        jd_text = [normalized_job_description_text]
        insert('Resume', texts)
        insert('JD', jd_text)
        resume = retrieve_point_by_id('Resume', 1)
        jd = retrieve_point_by_id('JD', 1)
        output_format = "('Numerical format of matching score in int type',[list of skills matched])"
        score, skills = eval(get_result(
            "You are an AI designed to assist an IT recruiter in evaluating resumes. Your task is to calculate the matching score out of 100(dont add extra information) and identify a list of matched skills between a normalized resume text and a normalized job description (JD) text. The matching score should be a numerical representation of the similarity between the two texts, and the list of matched skills should include all skills that appear in both texts. Return the results in a 'tuple format' without any additional explanation.",
            f'These are the resume and JD: {normalized_resume_text} and {normalized_job_description_text}."Maintain this form {output_format} as output"'
        ))
        st.subheader(f"Match Score: {score}%")
        st.write("### Matched Skills:")
        st.write(skills)
    else:
        st.warning("Please upload a resume and input a job description.")

def feedback():
    st.title("💬 Feedback")
    feedback = st.text_area("Your feedback", height=200, label_visibility='collapsed')
    if st.button("Submit"):
        st.success("Thank you for your feedback!")

# Navigation
st.sidebar.title("Navigation")
if st.sidebar.button("Home"):
    st.session_state.page = 'home'
if st.sidebar.button("About"):
    st.session_state.page = 'about'
if st.sidebar.button("Contact"):
    st.session_state.page = 'contact'

# Page routing
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if st.session_state.page == 'home':
    homepage()
elif st.session_state.page == 'upload_resume':
    upload_resume()
elif st.session_state.page == 'input_jd':
    input_jd()
elif st.session_state.page == 'results':
    results()
elif st.session_state.page == 'feedback':
    feedback()
elif st.session_state.page == 'about':
    st.write("About the service...")
elif st.session_state.page == 'contact':
    st.write("Contact information...")

# Mock function for extracting text from uploaded files
def extract_text_from_file(uploaded_file):
    return read_pdf(uploaded_file)