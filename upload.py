import os   
# from dotenv import load_dotenv
import fitz  # PyMuPDF for reading PDF files
from openai import OpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
# import streamlit as st
import re
from config import OPEN_API_KEY
import ast

OPENAI_API_KEY = OPEN_API_KEY
client1 = OpenAI(api_key=OPEN_API_KEY)
client = QdrantClient(url="http://localhost:6333", timeout=60.0)

def read_pdf(file_path):
    text = ""
    try:
        pdf_document = fitz.open(file_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        # st.error(f"Error reading {file_path}: {e}")
        print(e)
        exit(1)
    return text

def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', cleaned_text)
    return cleaned_text.strip()

def extract_information_gpt35_turbo(text, prompt, temperature=0.1):
    response = client1.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant helping me in the project."},
            {"role": "user", "content": prompt + text}
        ],
        temperature = temperature
    )
    return response.choices[0].message.content

def extract_experience(text):
    prompt = "Extract the experience details from the following text, don't add extra info:\n\n"
    return extract_information_gpt35_turbo(text, prompt)

def extract_education(text):
    prompt = "Extract the education details from the following text and don't add extra info:\n\n"
    return extract_information_gpt35_turbo(text, prompt)

def extract_skills(text):
    prompt = "Extract the technical skills only from the following text and don't add extra info:\n\n"
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

    prompt = f"I want you to give only percentage matching score between resume and job description on the basis of title, skills, education, experience whose details are as follows: resume title {res_title} with job title {job_title} resume skills which is: {res_skills} with job description skills: {job_skills}, resume experience: {res_exp} with job description experience: {job_exp} and finally resume education: {res_edu} with job description education: {job_edu}. And give answer ONLY and ONLY in the form Job title : Job title name | Similarity score: Answer in percentage."
    text = "You are helping me in project and supposed to give numerical values always."
    return extract_information_gpt35_turbo(text, prompt)

def create_collection(collection_name, vector_size):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
    )

def store_in_qdrant(job_descriptions):    
    points = []
    # if not client.collection_exists(collection_name="job_des_list"):
    #     create_collection("job_des_list",10)
    for i, description in enumerate(job_descriptions, start=1):
        points.append(models.PointStruct(id=i, payload={"description": description}, vector=[0.0]))
    client.upsert(
        collection_name="job_des_list",
        points=points
    )

def fetch_job_description(point_id):    
    point_text = client.retrieve(
        collection_name="job_des_list",
        ids=[point_id]
    )
    return point_text
def store_in_qdrant_single(job_description):    
    points=models.PointStruct(id=1, payload={"description": job_description}, vector=[0.0])
    client.upsert(
        collection_name="job_des_list",
        points=[points]
    )

i=1
if not client.collection_exists('job_des_list'):
    create_collection('job_des_list', 1)
jd_files=[i for i in os.listdir(r'D:\codespace\AIResumeMatchMaker\JD') if i.endswith('.pdf')]
jd_text=read_pdf('./jd/AI_Jd.pdf')
store_in_qdrant_single(jd_text)
# while i<10:
#     jd_text = []
#     for jd in jd_files:
#         jdt = read_pdf('./jd/'+jd)
#         jd_clean = clean_text(jdt)
#         jd_text.append(jd_clean)

    
    # store_in_qdrant(jd_text)