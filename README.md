# AI Resume Evaluator Pro

## Overview
AI Resume Evaluator Pro is a powerful resume analysis tool that evaluates resumes based on format, content, job relevance, ATS compatibility, and clarity. It provides actionable feedback and recommendations to improve job applications.

## Features
- Extracts text from PDF and DOCX resumes.
- Analyzes resume structure, content, and achievements.
- Evaluates ATS (Applicant Tracking System) compatibility.
- Matches resumes against job descriptions.
- Provides personalized feedback based on job title and industry.
- Offers role-specific recommendations for improvement.

## Installation

### **1. Clone the Repository**
```bash
git clone [https://github.com/your-repo/ai-resume-evaluator.git](https://github.com/MoeHamzaA/resume_AI.git)
cd ai-resume-evaluator
```

### **2. Set Up a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### **3. Install Dependencies**
Install all required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

If you need to generate a `requirements.txt` file manually, run:
```bash
pip freeze > requirements.txt
```

## Required Dependencies

The project relies on the following Python libraries:
- `gradio` – UI framework for building interactive applications.
- `pdfplumber` – Extracts text from PDF files.
- `python-docx` – Reads DOCX files.
- `re` – Regular expressions for text processing.
- `torch` – Deep learning framework for NLP models.
- `transformers` – Hugging Face's library for using pre-trained NLP models.
- `spacy` – NLP processing (requires `en_core_web_sm` model).
- `nltk` – Text processing and stopword removal.
- `time` – Used for measuring model load times.

To ensure all necessary packages are installed, use:
```bash
pip install gradio pdfplumber python-docx torch transformers spacy nltk
```

Additionally, download required NLP models:
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Usage

### **Running the Application**
Once dependencies are installed, run the application using:
```bash
python app.py
```
This will launch the Gradio interface in your web browser where you can upload a resume and receive AI-driven feedback.

### **Uploading and Analyzing Resumes**
1. Upload a resume in **PDF or DOCX** format.
2. Enter the target job title.
3. (Optional) Add a job description for better keyword matching.
4. Click **"Analyze My Resume"**.
5. Review the feedback, including ATS score, strengths, and areas for improvement.

## Troubleshooting

### **Common Issues & Solutions**
- **Error: `ModuleNotFoundError`**  
  Ensure all dependencies are installed:
  ```bash
  pip install -r requirements.txt
  ```

- **Error: `OSError: [E050] Can't find model 'en_core_web_sm'`**  
  Download the required NLP model:
  ```bash
  python -m spacy download en_core_web_sm
  ```

- **Error: `nltk` stopwords missing**  
  Download required `nltk` resources:
  ```bash
  python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
  ```

## License
This project is open-source and available under the MIT License.
