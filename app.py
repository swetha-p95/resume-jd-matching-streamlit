
import streamlit as st
import pdfplumber
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
from model import extract_skills_resume, extract_skills_jd  , sw_semantic_similarity_from_bert 

# Streamlit app
st.title("Resume and Job Description Matching Score with Customized NER")


# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')

def pdf_to_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def clean(text):
    """
    Clean the input text by removing URLs, emails, special characters, and stop words.

    :param text: The string to be cleaned
    :return: The cleaned string
    """
    # Compile patterns for URLs and emails to speed up cleaning process
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    # Remove URLs
    clean_text = url_pattern.sub('', text)

    # Remove emails
    clean_text = email_pattern.sub('', clean_text)

    # Remove special characters (keeping only words and whitespace)
    clean_text = re.sub(r'[^\w\s]', '', clean_text)

    # Remove stop words by filtering the split words of the text
    stop_words = set(stopwords.words('english'))
    clean_text = ' '.join(word for word in clean_text.split() if word.lower() not in stop_words)

    return clean_text

st.header("Upload Resume as a PDF file")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")




st.write("### Enter Job Description Text")
job_description = st.text_area("Job Description", height=200)





if st.button("Check Match Score"):
    
    if uploaded_file is not None:
     try:
        with st.spinner('Extracting resume text from PDF...'):
            text = pdf_to_text(uploaded_file)
            #st.success('Text extraction successful!')

            

            with st.spinner('Cleaning text...'):
                cleaned_text = clean(text)
                resume = cleaned_text
                st.success('Resume text cleaning successful!')

            
     except Exception as e:
        st.error(f"An error occurred: {e}")
    if job_description is not None:
     try:
        with st.spinner('Cleaning job description text...'):
            cleaned_job_description = clean(job_description)
            jd = cleaned_job_description 

            
            st.success('Job description text cleaning successful!')

        
     except Exception as e:
        st.error(f"An error occurred: {e}")

    if resume and jd:
        # Extract skills
        resume_skills = extract_skills_resume(resume)
        if resume_skills:
          st.success('Extracting skills from resume successful!')
        else:
          st.error('No skills extracted from resume')
        jd_skills = extract_skills_jd(jd)
        if jd_skills:
          st.success('Extracting skills from job description successful!')
        else:
          st.error('No skills extracted from job description')
        # Display extracted skills
        # if resume_skills:
        #     st.write("**Resume Skills:**")
        #     st.write(", ".join(resume_skills))
        # if jd_skills:
        #     st.write("**Job Description Skills:**")
        #     st.write(", ".join(jd_skills))
        
        # Check match using BERT
        if resume_skills and jd_skills:
            score, sim_count, match_count = sw_semantic_similarity_from_bert(jd_skills, resume_skills)
            if score >=0.5:
                st.success('More than 50% Match')
            else:
                st.error('Less than 50% Match')
            st.write(f"Matching Score: {score:.3f}")
            st.write(f"Number of Exact Matches between Resume and Job Description: {match_count}")
            st.write(f"Number of Similarity Matches between Resume and Job Description: {sim_count}")
            # Plot the pie chart based on matching score
            labels = 'Match Percentage', 'Non-Match Percentage'
            sizes = [score, 1 - score]  # Sizes based on matching score
            colors = ['#2ca02c', '#d62728']  # Green and red colors
            explode = (0.1, 0)  # Explode the first slice
            
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            
            st.pyplot(fig1)
    else:
        st.write("Please provide both resume and job description.")
