import streamlit as st
from transformers import pipeline
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# Page config
st.set_page_config(page_title="AI Resume Screener", page_icon="💼")
st.title("💼 AI Resume Screener")
st.write("Upload resumes and enter a job description — AI will rank candidates automatically!")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except:
        return ""

# Function to calculate match score
def calculate_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(score * 100, 2)
    except:
        return 0.0

# Load summarizer
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", 
                   model="sshleifer/distilbart-cnn-12-6")

# Job description input
st.subheader("📝 Step 1 — Enter Job Description")
job_description = st.text_area(
    "Paste the job description here:",
    placeholder="""Example:
We are looking for a Python Developer with experience in:
- Machine Learning and AI
- TensorFlow or PyTorch
- REST APIs and Flask
- SQL databases
- Strong communication skills""",
    height=200
)

# Resume upload
st.subheader("📁 Step 2 — Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload PDF resumes (you can select multiple!)",
    type=["pdf"],
    accept_multiple_files=True
)

# Analyze button
if st.button("🔍 Screen Resumes"):
    if not job_description.strip():
        st.warning("⚠️ Please enter a job description first!")
    elif not uploaded_files:
        st.warning("⚠️ Please upload at least one resume!")
    else:
        results = []
        
        with st.spinner("Analyzing resumes..."):
            for pdf_file in uploaded_files:
                # Extract text
                resume_text = extract_text_from_pdf(pdf_file)
                
                if resume_text.strip():
                    # Calculate match score
                    score = calculate_match_score(resume_text, job_description)
                    
                    # Get candidate name from filename
                    candidate_name = pdf_file.name.replace(".pdf", "").replace("_", " ").title()
                    
                    results.append({
                        "Candidate": candidate_name,
                        "Match Score": f"{score}%",
                        "Score Value": score,
                        "Status": "✅ Strong Match" if score >= 50 
                                 else "⚠️ Moderate Match" if score >= 30 
                                 else "❌ Weak Match"
                    })
                else:
                    st.error(f"Could not read {pdf_file.name} — make sure it's a text-based PDF!")

        if results:
            # Sort by score
            results_sorted = sorted(results, key=lambda x: x["Score Value"], reverse=True)
            
            # Show ranked results
            st.subheader("🏆 Ranking Results")
            
            for i, candidate in enumerate(results_sorted):
                rank = i + 1
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
                
                col1, col2, col3 = st.columns([1, 3, 2])
                with col1:
                    st.metric("Rank", medal)
                with col2:
                    st.metric("Candidate", candidate["Candidate"])
                with col3:
                    st.metric("Match Score", candidate["Match Score"])
                
                st.write(f"Status: {candidate['Status']}")
                st.divider()
            
            # Summary table
            st.subheader("📊 Summary Table")
            df = pd.DataFrame([{
                "Rank": i+1,
                "Candidate": r["Candidate"],
                "Match Score": r["Match Score"],
                "Status": r["Status"]
            } for i, r in enumerate(results_sorted)])
            
            st.dataframe(df, use_container_width=True)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="resume_screening_results.csv",
                mime="text/csv"
            )