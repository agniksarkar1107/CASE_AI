# CASE_AI
# Legal AI RAG - Legal Document Analyzer & Case Outcome Prediction

## 📌 Project Overview
The **Legal AI RAG (Retrieval-Augmented Generation) System** is an AI-powered tool designed to analyze legal documents, extract relevant legal precedents, and predict case outcomes using advanced NLP and machine learning models. The system leverages:
- **Streamlit** for an interactive UI
- **FAISS/Pinecone** for vector-based legal document retrieval
- **Google Gemini AI** for in-depth legal analysis & predictions
- **OCR (Tesseract)** for extracting text from images/PDFs
- **Web Scraping (BeautifulSoup)** for gathering relevant legal precedents from online sources

---

## 🚀 Features
✅ **Legal Document Parsing**: Extracts text from PDFs and images
✅ **Semantic Search on Legal Cases**: Retrieves past legal precedents using FAISS/Pinecone
✅ **Case Outcome Prediction**: Uses Gemini AI to predict potential case outcomes with probability estimates
✅ **Legal Precedent Summarization**: AI-generated structured summaries of relevant case laws
✅ **Risk & Strategy Analysis**: Identifies risks and suggests legal strategies for the case
✅ **Downloadable Reports**: Generates AI-processed case analysis for easy reference

---

## 🛠️ Tech Stack
- **Programming Language**: Python
- **Frameworks**: Streamlit, LangChain
- **AI/ML**: Google Gemini API
- **Vector DB**: Pinecone
- **NLP**: spaCy
- **OCR**: Tesseract
- **Web Scraping**: BeautifulSoup
- **Data Processing**: NumPy, OpenCV

---

## 📦 Installation
To set up the project, follow these steps:

### **1️⃣ Clone the Repository**
```sh
 git clone https://github.com/agniksarkar1107/CASE_AI.git
 cd CASE_AI
```

### **2️⃣ Install Dependencies**
```sh
 pip install -r requirements.txt
```

### **3️⃣ Set Up API Keys**
Create a `.env` file and add your API keys:
```env
PINECONE_API_KEY=your_pinecone_key
GEMINI_API_KEY=your_gemini_key
```

### **4️⃣ Run the Streamlit App**
```sh
 streamlit run app.py
```

---

## 📄 Usage Guide
1️⃣ **Upload a Legal Document (PDF/Image)**
- The system extracts text and identifies legal terms.

2️⃣ **Retrieve Relevant Legal Precedents**
- Scrapes Indian case law sources and structures them neatly.

3️⃣ **Predict Case Outcomes**
- AI generates probability-based legal predictions with risk analysis.

4️⃣ **Download Legal Report**
- Save the AI-generated legal analysis for further reference.

---

## 📝 Requirements (requirements.txt)
```
streamlit
spacy
requests
pypdf2
pytesseract
opencv-python-headless
numpy
beautifulsoup4
pinecone-client
langchain
langchain-google-genai
torch
sentence-transformers
dotenv
```

---

## 🤖 Future Enhancements
🔹 **Integration with Indian Supreme Court API for real-time legal data**  
🔹 **Support for multilingual legal document analysis**  
🔹 **More advanced legal reasoning using fine-tuned LLMs**  

---

## 🤝 Contribution
Want to contribute? Fork the repository, create a branch, and submit a PR!

---

## 🏛️ Legal Disclaimer
This AI system is **not a substitute for professional legal advice**. Always consult a licensed attorney for legal matters.

---

## 📬 Contact
🔗 **GitHub**: [your-repo-link](https://github.com/yourusername/legal-ai-rag)  
✉️ **Email**: your-email@example.com  
📢 **LinkedIn**: [your-profile](https://linkedin.com/in/yourprofile)


