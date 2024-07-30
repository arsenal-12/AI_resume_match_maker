import os
from dotenv import load_dotenv
import fitz  # PyMuPDF for reading PDF files
from openai import OpenAI
import streamlit as st
# Set your OpenAI API key
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client1 = OpenAI()

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def read_pdf(file_path):
    text = ""
    try:
        pdf_document = fitz.open(file_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text

def extract_information_gpt35_turbo(text, prompt, temperature=0.7):
    response = client1.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant helping me in my project."},
            {"role": "user", "content": prompt + text}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content


def extract_experience(text):
    prompt = "Extract the experience details from the following text, don't add extra info:\n\n"
    return extract_information_gpt35_turbo(text, prompt)

def extract_education(text):
    prompt = "Extract the education details from the following text, don't add extra info:\n\n"
    return extract_information_gpt35_turbo(text, prompt)

def extract_skills(text):
    prompt = "Extract the technical skills only from the following text, don't add extra info:\n\n"
    return extract_information_gpt35_turbo(text, prompt)

def extract_title(text):
    prompt = "Extract the job title or resume title only from the following text and don't add extra info:\n\n"
    return extract_information_gpt35_turbo(text, prompt)

def similarity(resume, job_desc):
    res_title = extract_title(resume)
    res_exp = extract_experience(resume)
    res_edu = extract_education(resume)
    res_skills = extract_skills(resume)

    job_title = extract_title(job_desc)
    job_exp = extract_experience(job_desc)
    job_edu = extract_education(job_desc)
    job_skills = extract_skills(job_desc)


    prompt = f"I want you to give only percentage matching score between resume and job description on the basis of title, skills, education, experience whose details are as follows: resume title: {res_title} with job description title: {job_title}, resume skills which is: {res_skills} with job description skills: {job_skills}, resume experience: {res_exp} with job description experience: {job_exp} and finally resume education: {res_edu} with job description education: {job_edu}. And give answer ONLY and ONLY in the form 'Similarity score: Answer in percentage' and on next line give the reason in ONLY one line. Keep in mind the job title and resume title also."
    text = "You are helping me in project and supposed to give numerical values."
    return extract_information_gpt35_turbo(text, prompt)

# Streamlit app
st.markdown('<h1 class="playwrite-it-moderna-custom">AI Resume Match Maker</h1>', unsafe_allow_html=True)

st.markdown('<p class="playwrite-us-modern-custom">By Aanisha</p>', unsafe_allow_html=True)

# Custom CSS for enhancing style
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cookie&family=Playwrite+IT+Moderna:wght@100..400&display=swap');
     @import url('https://fonts.googleapis.com/css2?family=Cookie&family=Playwrite+IT+Moderna:wght@100..400&family=Playwrite+US+Modern:wght@100..400&display=swap');
    .cookie-regular {
        font-family: "Cookie", cursive;
        font-weight: 400;
        font-style: normal;
    }

    .playwrite-it-moderna-custom {
        font-family: "Playwrite IT Moderna", cursive;
        font-optical-sizing: auto;
        font-weight: 300; /* You can change the weight as needed */
        font-style: normal;
    }
    .playwrite-us-modern-custom {
    font-family: "Playwrite US Modern", cursive;
    font-optical-sizing: auto;
    font-weight: 400;
    font-style: normal;
    }
    .st-emotion-cache-1y4p8pa{
        background: linear-gradient(90deg in oklab, #fac2ca, #ed8ad9);
        border-radius: 15px;
        padding: 20px;
        margin: 20px auto;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        width: 100%;
    }
    .st-emotion-cache-1uixxvy{
    color: black;
    }
    #ai-resume-match-maker{
        color: rgb(10 1 46);
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
    }
    button{
        background-color: yellow;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .st-emotion-cache-q8sbsg p{
    font-size:1rem;
    text-align: center;
    font-size:1.5rem;
    color: rgb(10 1 46);
    font-family: Serif;
    font-weight: bold;
    }
    p{
    text-align: center;
    color: #641b28;
    font-size:1.5rem;
    color: rgb(10 1 46);
    font-family: Serif;
    font-weight: bold;
    }
    .st-emotion-cache-13k62yr{
    background: linear-gradient(90deg in oklab, purple, pink);
    }
    .st-emotion-cache-1avcm0n {
    background: linear-gradient(90deg in oklab, purple, pink);
    }
    #similarity-score {
        text-align: center;
        font-size: 1.5em;
        color: #641b28;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<p class="playwrite-us-modern-custom">Single file OpenAI approach</p>', unsafe_allow_html=True)
# Upload section
st.markdown('<p class="playwrite-us-modern-custom">Upload your Resume and Job description to find a match.</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader('Resume', type=['pdf', 'txt'])

with col2:
    job_desc_file = st.file_uploader('Job Description', type=['pdf', 'txt'])


if resume_file and job_desc_file:
    if resume_file.type == "application/pdf":
        resume_text = read_pdf(resume_file)
    else:
        resume_text = resume_file.read().decode('utf-8')

    if job_desc_file.type == "application/pdf":
        job_desc_text = read_pdf(job_desc_file)
    else:
        job_desc_text = job_desc_file.read().decode('utf-8')

    similarity_score = similarity(resume_text, job_desc_text)
    
    st.markdown(f'<div id="similarity-score">{similarity_score}</div>', unsafe_allow_html=True)