This repository contains the code for a Streamlit app which givessemantic similarity based matching scores between a resume and job description. 

For this we have trained two custom NER models that can extract skills from resumes and job descriptions respectively. The code for training the custom NER is also provided.

The extracted skills are then matched using a semantic similarity score obtained using BERT embeddings. Here, a score above 0.6 is considered a match.

The resume and jd datasets are provided. This dataset contains mostly data science resumes and job descriptions. The model can be improved by expanding the dataset. 
