import streamlit as st
import requests
import PyPDF2
import pytesseract
import cv2
import numpy as np
from bs4 import BeautifulSoup
import pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
  # Corrected Import
from langchain.chains import RetrievalQA

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from concurrent.futures import ThreadPoolExecutor

# Initialize Streamlit App
st.title("Legal Document Analyzer & Case Prediction AI")

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
        "https://www.barandbench.com/search?q="
    ]
    
    def fetch_data(url):
        response = requests.get(url + query)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    
    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_data, urls)
    
    return "\n".join(results)

# Initialize Pinecone
from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_2TJcwi_1Yuu9ZbDvRS7egDUx7nanbmBWpmfS2brogh6zSb9bNr2ifaR1m8tkjynGEQmmY")
index = pc.Index("legal-cases")

vectorstore = Pinecone(index, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


# Initialize AI agents
gemini_agent = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="AIzaSyBdz-qcLFRDsR-mm37AlRf2w6RZws2lDL0")

qa_chain = RetrievalQA(llm=gemini_agent, retriever=retriever)

# Upload Legal Document
uploaded_file = st.file_uploader("Upload a legal document (PDF/Image)", type=["pdf", "png", "jpg"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        document_text = extract_text_from_pdf(uploaded_file)
    else:
        document_text = extract_text_from_image(uploaded_file)
    
    st.subheader("Extracted Text")
    st.text_area("", document_text, height=300)
    
    # Scrape related legal cases
    scraped_text = scrape_legal_sources(document_text[:100])
    st.subheader("Relevant Legal Precedents")
    st.text_area("", scraped_text, height=300)
    
    # Predict Case Outcome
    case_prediction = qa_chain.run(document_text)
    st.subheader("Case Outcome Prediction")
    st.write(case_prediction)
    
    # Generate Report
    st.download_button("Download Report", case_prediction, file_name="case_analysis_report.txt")
