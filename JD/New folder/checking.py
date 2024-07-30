import re
import nltk
import spacy
import time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import pathlib
import textwrap
import google.generativeai as genai
from fpdf import FPDF
from main import read_pdf
nlp = spacy.load('en_core_web_sm')
def clean_text(text):
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove phone numbers (10 digits)
    text = re.sub(r'\b\d{10}\b', '', text)
    # Remove extra spaces
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
    text = text.lower()  # Convert to lowercase
    text = clean_text(text)  # Clean text
    tokens = tokenize_text(text)  # Tokenize text
    tokens = remove_stopwords(tokens)  # Remove stop words
    if use_stemming:
        tokens = stem_tokens(tokens)  # Stem tokens
    else:
        tokens = lemmatize_tokens(tokens)  # Lemmatize tokens
    temp= ' '.join(tokens) 
    return temp
def extract_resume_features(text):
    features = {}
    
    # Extract name (assuming name is the first line)
    name = text.split('\n')[0]
    features['name'] = name
    
    # Extract contact information
    email = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    phone = re.search(r'\b\d{10}\b', text)
    
    features['email'] = email.group(0) if email else None
    features['phone'] = phone.group(0) if phone else None
    
    # Extract skills (assuming skills are listed under a section titled "Skills")
    skills = re.search(r'skills(.*?)(experience|education|achievements|$)', text, re.DOTALL | re.IGNORECASE)
    if skills:
        skills = skills.group(1).strip().split('\n')
        skills = [skill.strip() for skill in skills if skill.strip()]
    features['skills'] = skills if skills else []

    # Extract experience (assuming experience is listed under a section titled "Experience")
    experience = re.search(r'experience(.*?)(skills|education|achievements|$)', text, re.DOTALL | re.IGNORECASE)
    if experience:
        experience = experience.group(1).strip().split('\n')
        experience = [exp.strip() for exp in experience if exp.strip()]
    features['experience'] = experience if experience else []

    # Extract education (assuming education is listed under a section titled "Education")
    education = re.search(r'education(.*?)(skills|experience|achievements|$)', text, re.DOTALL | re.IGNORECASE)
    if education:
        education = education.group(1).strip().split('\n')
        education = [edu.strip() for edu in education if edu.strip()]
    features['education'] = education if education else []

    # Extract achievements (assuming achievements are listed under a section titled "Achievements")
    achievements = re.search(r'achievements(.*?)(skills|experience|education|$)', text, re.DOTALL | re.IGNORECASE)
    if achievements:
        achievements = achievements.group(1).strip().split('\n')
        achievements = [ach.strip() for ach in achievements if ach.strip()]
    features['achievements'] = achievements if achievements else []

    return features

def extract_job_features(text):
    features = {}
    
    # Extract job title (assuming title is the first line)
    title = text.split('\n')[0]
    features['title'] = title
    
    # Extract responsibilities (assuming responsibilities are listed under a section titled "Responsibilities")
    responsibilities = re.search(r'responsibilities(.*?)(requirements|$)', text, re.DOTALL | re.IGNORECASE)
    if responsibilities:
        responsibilities = responsibilities.group(1).strip().split('\n')
        responsibilities = [resp.strip() for resp in responsibilities if resp.strip()]
    features['responsibilities'] = responsibilities if responsibilities else []

    # Extract requirements (assuming requirements are listed under a section titled "Requirements")
    requirements = re.search(r'requirements(.*?)(experience|$)', text, re.DOTALL | re.IGNORECASE)
    if requirements:
        requirements = requirements.group(1).strip().split('\n')
        requirements = [req.strip() for req in requirements if req.strip()]
    features['requirements'] = requirements if requirements else []

    return features
resume=read_pdf('TarunResume.pdf')
print(extract_resume_features(normalize_text(resume)))