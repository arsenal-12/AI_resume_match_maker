import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer, util
from qdrant_client import QdrantClient
from openai import OpenAI
import re
import spacy
import textwrap
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from config import OPEN_API_KEY
from docx import Document
import streamlit.components.v1 as components
# Initialize models and clients
nlp = spacy.load('en_core_web_sm')
model = SentenceTransformer('all-MiniLM-L6-v2')
open_client = OpenAI(api_key=OPEN_API_KEY)
qdrant_client = QdrantClient("http://localhost:6333", timeout=60.0)
st.set_page_config(
    page_title="AI Resume Match Maker",
    page_icon="https://raw.githubusercontent.com/tarun261003/PdfViewer/main/IS.png",
    layout="wide",
    initial_sidebar_state="auto",
)
def sorted_list(scores):
    prompt = f"So this is the list of similarity score matches: {scores}. I want you to ONLY sort each item based on similarity score, which is greater than 50%, in descending order and return it rank wise exactly as it was in input. Don't make any changes, just sort based on similarity score. Don't give anything else as output, not even acknowledgement."
    text = "You are helping me in project and ONLY returning values and nothing else."
    response = extract_information_gpt35_turbo(text, prompt)
    return response.strip().split('\n')

def clean_text(text):
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def lemmatize_tokens(tokens):
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]

def normalize_text(text, use_stemming=False):
    text = text.lower()
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = stem_tokens(tokens)
    else:
        tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)

def read_pdf(file):
    if file.type == "application/pdf":
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    else:
        raise ValueError("Unsupported file format. Please upload PDF, TXT, or DOCX files.")

    return text

def extract_information_gpt35_turbo(text, prompt, temperature=0.1):
    response = open_client.chat.completions.create(
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

def fetch_job_description(point_id):    
    point_text = qdrant_client.retrieve(
        collection_name="job_des_list",
        ids=[point_id]
    )
    return point_text

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
    text = '''You are helping me in project and supposed "to give numerical values such that it shoud be stored as integer." and dont give any extra information.'''
    return job_title, extract_information_gpt35_turbo(text, prompt)

# Streamlit UI
st.markdown("""
  <div style="display: flex; align-items: center;">
    <img src="https://raw.githubusercontent.com/tarun261003/PdfViewer/main/IS-removebg-preview.png" alt="Logo" style="width: 70px; height: 70px; margin-right: 10px;">
    <h1 style="text-align: center; color: #EEF7FF; margin-bottom: 0; font-family: 'Open Sans', sans-serif;">AI ResumeMatch Maker</h1>
  </div>
""", unsafe_allow_html=True)

option = st.radio("Choose Matcher", ("One Resume and One Job Description Matcher", "One Resume and Multiple Job Descriptions Matcher"))

if option == "One Resume and One Job Description Matcher":
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf","docx","txt"], accept_multiple_files=False)
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf","docx","txt"], accept_multiple_files=False)

    if resume_file is not None and jd_file is not None:
        resume_text = read_pdf(resume_file)
        job_desc_text = read_pdf(jd_file)

        col1, col2 = st.columns(2)

        with col1:
            resume_text = st.text_area("Resume Text", value=resume_text, height=300,help="No need to remove white spaces and special charecters.We will handle them.")

        with col2:
            job_desc_text = st.text_area("Job Description Text", value=job_desc_text, height=300,help="No need to remove white spaces and special charecters.We will handle them.")

        if st.button("Check Match"):
            with st.spinner('Checking resume'):
                jd_title, similarity_score = similarity(resume_text, job_desc_text)
                score_match = re.search(r'Similarity score: (\d+)%', similarity_score)
                sim = int(score_match.group(1)) if score_match else None
                reason_match = re.search(r'Reason: (.+)', similarity_score)
                reason = reason_match.group(1) if reason_match else None
                
                if sim < 25:
                    st.toast('No Match',icon="🚫")
                    st.markdown(f'<div id="similarity-score" style="color:red;">JobTitle: {jd_title}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div id="similarity-score" style="color:red;">Similarity: {sim}%</div>', unsafe_allow_html=True)
                    st.markdown(f'<div id="similarity-score" style="color:red;">{reason}</div>', unsafe_allow_html=True)
                elif sim >= 25 and sim < 50:
                    st.toast('Low Match',icon="🪫")
                    st.markdown(f'<div id="similarity-score" style="color:#F2FBFB;">Low Match Rate</div>', unsafe_allow_html=True)
                    st.markdown(f'<div id="similarity-score" style="color:#F2FBFB;">JobTitle: {jd_title}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div id="similarity-score" style="color:#F2FBFB;">Similarity: {sim}%</div>', unsafe_allow_html=True)
                    st.markdown(f'<div id="similarity-score" style="color:#F2FBFB;">{reason}</div>', unsafe_allow_html=True)
                elif sim >= 50 and sim < 75:
                    st.toast('Average Match',icon='🧑‍💻')
                    st.markdown(f'<div id="similarity-score" style="color:#F4CD7A;">Average Match Rate</div>', unsafe_allow_html=True)
                    st.markdown(f'<div id="similarity-score" style="color:#F4CD7A;">JobTitle: {jd_title}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div id="similarity-score" style="color:#F4CD7A;">Similarity: {sim}%</div>', unsafe_allow_html=True)
                    st.markdown(f'<div id="similarity-score" style="color:#F4CD7A;">{reason}</div>', unsafe_allow_html=True)
                elif sim >= 75:
                    st.toast('Matched',icon='✅')
                    st.markdown(f'<div id="similarity-score" style="color:lime;">JobTitle: {jd_title}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div id="similarity-score" style="color:lime;">Similarity: {sim}%</div>', unsafe_allow_html=True)
                    st.markdown(f'<div id="similarity-score" style="color:lime;">{reason}</div>', unsafe_allow_html=True)
    else:
        if resume_file is not None and jd_file is None:
            st.write("Please upload a Job description in PDF format.")
        elif resume_file is None and jd_file is not None:
            st.write("Please upload a Resume in PDF format.")
        else:
            st.write("Please upload both a resume and a job description in PDF format.")

elif option == "One Resume and Multiple Job Descriptions Matcher":
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf","docx","txt"], accept_multiple_files=False)
    if resume_file is not None:
        resume_text = read_pdf(resume_file)
        resume_text = st.text_area("Resume Text", value=resume_text, height=300,help="No need to remove white spaces and special charecters.We will handle them")

        if st.button("Check Match"):
            with st.spinner('Checking resume'):
                normalized_resume_text = normalize_text(resume_text)
                scores = []
                for i in range(1,11):
                    jdq_text = fetch_job_description(i)
                    a = str(jdq_text[0].payload.values())
                    jd_title, similarity_score = similarity(normalized_resume_text, a)
                    score_match = re.search(r'Similarity score: (\d+)%', similarity_score)
                    sim = int(score_match.group(1)) if score_match else None
                    reason_match = re.search(r'Reason: (.+)', similarity_score)
                    reason = reason_match.group(1) if reason_match else None
                    scores.append([jd_title, sim])

                try:
                    sorted_score_list = sorted(scores, key=lambda x: x[1], reverse=True)
                except Exception as e:
                    print(scores)
                
                if sorted_score_list[0][1]>=50:
                    st.markdown(f'<div id="similarity-score" style="font-size:25px;">Top match:</div>', unsafe_allow_html=True)
                    st.markdown(f'<div id="similarity-score" style="font-size:20px; margin-left:20px;">JobTitle:Top Match:</div>', unsafe_allow_html=True)
                    st.markdown(f'<div id="similarity-score" style="font-size:20px; margin-left:20px;">JobTitle: {sorted_score_list[0][0]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div id="similarity-score" style="font-size:20px; margin-left:20px;">Similarity Score: {sorted_score_list[0][1]}%</div>', unsafe_allow_html=True)
                    sorted_score_list.pop(0)
                    st.markdown('''<hr style="border-style: dotted;border-bottom: none;border-color:#EEF7FF ;border-width: 4px;width: 4%;"/>''',unsafe_allow_html=True)
                for i in sorted_score_list:
                    if i[1] > 40:
                        st.markdown(f'<div id="similarity-score">JobTitle: {i[0]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div id="similarity-score">Similarity Score: {i[1]}%</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div id="similarity-score">{i[0]} - No Match</div>', unsafe_allow_html=True)
    else:
        st.write("Please upload a resume in PDF format.")
