import os

import streamlit as st
import re

import requests
import PyPDF2
import pytesseract
import cv2
import numpy as np
from bs4 import BeautifulSoup
import pinecone
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
  # Corrected Import
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone as PineconeVectorStore

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from concurrent.futures import ThreadPoolExecutor

#custom css
def custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&family=Playfair+Display:wght@400;700&family=Nunito:wght@300;400;700&display=swap');

        /* Background Gradient */
        html, body, [class*="st-emotion-cache"] {
            background: linear-gradient(to right, #E0EAFC, #CFDEF3);
            color: #2C3E50; /* Dark Blue Text */
            font-family: 'Poppins', sans-serif;
        }

        /* Main Heading */
        .main-heading {
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            color: #1A1A2E; /* Deep Blue */
            font-family: 'Playfair Display', serif;
            margin-top: 20px;
        }

        /* Subtitle */
        .sub-heading {
            text-align: center;
            font-size: 24px;
            font-weight: 400;
            color: #4A4A4A; /* Elegant Dark Grey */
            font-family: 'Nunito', sans-serif;
            margin-bottom: 20px;
        }

        /* Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #A1C4FD, #C2E9FB);
            color: white;
            font-size: 18px;
            padding: 12px 24px;
            border-radius: 12px;
            font-family: 'Nunito', sans-serif;
            border: none;
            transition: all 0.3s ease-in-out;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.15);
        }

        div.stButton > button:hover {
            background: linear-gradient(135deg, #C2E9FB, #A1C4FD);
            color: #1A1A2E;
            transform: scale(1.05);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 12px;
            font-family: 'Poppins', sans-serif;
            color: #1A1A2E;
        }

        /* Footer */
        .footer {
            position: fixed;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 14px;
            font-family: 'Nunito', sans-serif;
            color: #4A4A4A;
        }
        </style>
    """, unsafe_allow_html=True)

# Apply Custom Styles
custom_css()
import spacy
from spacy.cli import download
try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

# Initialize Streamlit App
st.title("CASE AI- A MULTIMODAL LEGAL RAG BY AGNIK SARKAR")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Function to extract text from image
def extract_text_from_image(image_file):
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    text = pytesseract.image_to_string(image)
    return text

# Function to scrape multiple Indian legal sources
def scrape_legal_sources(query):
    urls = [
        "https://www.indiankanoon.org/search/?formInput=", 
        "https://www.scconline.com/?s=", 
        "https://www.barandbench.com/search?q=",
        "https://www.livelaw.in/search?q=", 
        "https://www.thehindu.com/topic/legal-affairs/",
        "https://www.legallyindia.com/"
    ]
    
    def fetch_data(url):
        response = requests.get(url + query)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text and clean up spacing
        text = soup.get_text(separator="\n").strip()
        
        # Remove unwanted text patterns (navigation menus, repeated phrases)
        text = "\n".join(line for line in text.splitlines() if 20 < len(line.strip()) < 500)

        return text

    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_data, urls)
    
    combined_text = "\n\n".join(results)

    
    doc = nlp(combined_text)
    case_names = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    court_names = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    legal_sections = re.findall(r"Section \d+ of the [A-Za-z ]+ Act", combined_text)

    
    summary_prompt = f"""
    Summarize the following legal precedents into structured case summaries.
    Extract case name, court, date, legal section, and judgment:
    
    {combined_text}
    """
    structured_summary = gemini_agent.invoke(summary_prompt)
    


    structured_summary_text = structured_summary.content if hasattr(structured_summary, 'content') else str(structured_summary)




   
    formatted_output = "📌**Legal Case Precedents Summary**\n\n"
    for i, case in enumerate(structured_summary_text.split("\n\n")):
        formatted_output += f"""
        **{i+1}. Case Name**: {case_names[i] if i < len(case_names) else "N/A"}
        - **Court**: {court_names[i] if i < len(court_names) else "N/A"}
        - **Legal Section**: {legal_sections[i] if i < len(legal_sections) else "N/A"}
        - **Judgment Summary**: {case}
        ---
        """

    return formatted_output



# Initialize Pinecone
from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=PINECONE_API_KEY)
print(pc.list_indexes())






# Create Pinecone index if it does not exist
index_name = "legal-cases"
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
# Connect to the index
index = pc.Index(index_name)
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
retriever = PineconeVectorStore(index, gemini_embeddings, text_key="text").as_retriever()







# Initialize AI agents
gemini_agent = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=gemini_agent,
    retriever=retriever,
    chain_type="stuff"
)

# Upload Legal Document
uploaded_file = st.file_uploader("Upload a legal document (PDF/Image)", type=["pdf", "png", "jpg"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        document_text = extract_text_from_pdf(uploaded_file)
    else:
        document_text = extract_text_from_image(uploaded_file)
    
    st.subheader("Extracted Text")
    st.text_area("", document_text, height=200)
    
    # Scrape related legal cases
    

    st.subheader("Relevant Legal Precedents")
    structured_text = scrape_legal_sources(document_text[:200])  # Increased query length
    st.markdown(structured_text, unsafe_allow_html=True)
    
    
    case_analysis_prompt = f"""
    Provide a detailed legal analysis for the following case:
    
    **Case Details:** {document_text}
    
    **Required Analysis:**
    - Identify key legal issues.
    - Compare this case with similar past rulings.
    - Outline possible case outcomes with probability estimates.
    - Highlight risks and weaknesses in arguments.
    - Recommend a legal strategy for the best outcome.
    """
    case_analysis = gemini_agent.invoke(case_analysis_prompt)
    case_analysis_text = case_analysis.content if hasattr(case_analysis, 'content') else str(case_analysis)
    
    st.subheader("Detailed Case Analysis")
    st.text_area("", case_analysis_text, height=500)
    st.download_button("Download Case Analysis", case_analysis_text, file_name="case_detailed_analysis_report.txt")
  # Predict Case Outcome
    chat_history = []  # Initialize empty chat history

    case_prediction = qa_chain.run({"question": document_text, "chat_history": chat_history})
    st.subheader("Case Outcome")
    st.text_area("", case_prediction, height=100)

    
    # Generate Report
    st.download_button("Download Report", case_prediction, file_name="case_outcome_report.txt")
