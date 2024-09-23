from flask import Flask, render_template, request, redirect, url_for, session
import google.generativeai as genai
import fitz
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
import ast
import json
import psutil

app = Flask(__name__)
application = app

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))


# Load your job and book data
df = pd.read_excel('data/job_embed.xlsx')
df_book = pd.read_csv('data/ebook_emb.csv')

# Load Sentence Transformer model
e5 = SentenceTransformer('intfloat/e5-small-v2')

# Google generative AI configuration
generation_config = {
    "temperature": 0.1,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 500
}

safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Initialize generative model
model = genai.GenerativeModel(
    model_name="models/gemini-pro",
    generation_config=generation_config,
    safety_settings=safety_settings
)


# Route for the resume processing
@app.route('/', methods=['POST', 'GET'])
def resume():
    if request.method == 'POST':
        # Handle file upload separately
        file = validate_file(request)
        if not file:
            return "No valid file uploaded"

        # Extract resume details
        name, current_job, years_of_experience, education, skills, certification, desired_job_title, location = extract_resume_details(file)
        description_text = desired_job_title + skills

        # Solve for matching job title and company
        job_title, company_name = solve_job(description_text)

        # Check for book match and return the relevant information
        if name in df_book['fb_username'].to_list():
            book_name, book_author, book_description, book_id = solve_book(name)
        else:
            return render_resume_template(name, current_job, years_of_experience, education, skills, certification, desired_job_title, location, job_title, company_name, book_name=None, book_author=None, book_description=None, book_id=None)

        # Render results to template
        return render_resume_template(name, current_job, years_of_experience, education, skills, certification, desired_job_title, location, job_title, company_name, book_name, book_author, book_description, book_id)

    # Render default template
    return render_template('index.html')


def validate_file(request):
    """
    Validates if the uploaded file exists and is a PDF.
    """
    if 'resume' not in request.files:
        return None
    file = request.files['resume']
    if file.filename == '' or not file.filename.endswith('.pdf'):
        return None
    return file


def extract_resume_details(file):
    """
    Extract text from the uploaded PDF and generate key details.
    """
    # Extract content from PDF
    pdf_content = extract_text_from_pdf(file)

    # Use generative model to extract key details from the resume
    info = model.generate_content(f"""Screen this resume: {pdf_content}. List: name (just write fullname example: 
                                        Nguyen Anh Tuan, It's usually at the head of resume, capilize first letter), current job (under 3 words), years of experience (just 1 number), 
                                        education (High School Diploma or Bachelor's degree or Master degree), skills (just 2 skills), certificate (just 1 certificates), 
                                        desired job title, location (just show country). Every single feature separated by comma. Example: Kevin Murphy, Data scientist, 4, Bachelor's degree, 
                                        Python - R - PowerBI, Google Data Analytics, Data Analyst, Vietnam. Don't format anything please. If don't have anything type None. Just write as the example format.""")

    # Parse the response from the generative model
    return parse_resume_info(info.text)


def extract_text_from_pdf(pdf_file):
    """
    Extracts the text content from the PDF file.
    """
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text


def parse_resume_info(info_text):
    """
    Parse the resume details from the generative model's response.
    """
    parts = info_text.split(', ')
    name = parts[0]
    current_job = parts[1]
    years_of_experience = parts[2]
    education = parts[3]
    skills = parts[4]
    certification = parts[5]
    desired_job_title = parts[6]
    location = parts[7]
    return name, current_job, years_of_experience, education, skills, certification, desired_job_title, location


def solve_job(cv):
    """
    Find job matches based on the resume content using cosine similarity.
    """
    compare = pd.DataFrame(columns=['job_id', 'company_name', 'title', 'cosine_similarity_gte'])
    cv_embedded = e5.encode(cv)

    for i in range(len(df)):
        vector_str = df['all_e5_small_embedded'][i]
        vector_list = ast.literal_eval(vector_str)
        tensor = torch.tensor(vector_list)
        cosine_similarity = cos_sim(cv_embedded, tensor).item()

        compare = compare._append({
            'job_id': df['job_id'][i],
            'company_name': df['company_name'][i],
            'title': df['title'][i],
            'cosine_similarity': cosine_similarity
        }, ignore_index=True)

    compare = compare.sort_values(by='cosine_similarity', ascending=False)
    finish_compare = compare.head(4)
    job_title = finish_compare["title"].to_list()
    company_name = finish_compare["company_name"].to_list()
    return job_title, company_name


def solve_book(name):
    """
    Retrieve book information based on the user's name.
    """
    book = df_book[df_book['fb_username'] == name]
    book_name = book['book_name'].tolist()
    book_author = book['book_author'].tolist()
    book_description = book['book_description'].tolist()
    book_id = book['book_id'].tolist()
    for i in range(len(book_id)):
        book_id[i] = str(book_id[i])
    return book_name, book_author, book_description, book_id


def render_resume_template(name, current_job, years_of_experience, education, skills, certification, desired_job_title, location, job_title, company_name, book_name, book_author, book_description, book_id):
    """
    Render the final template with the extracted information.
    """
    return render_template(
        "index.html",
        name=name, current_job=current_job, years_of_experience=years_of_experience,
        education=education, skills=skills, certification=certification,
        desired_job_title=desired_job_title, location=location,
        job_title=job_title, company_name=company_name,
        book_name=book_name, book_author=book_author,
        book_description=book_description, book_id=book_id
    )

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")

if __name__ == '__main__':
    app.run(debug=True)