import gradio as gr
import pdfplumber
import docx
import re
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import spacy
import nltk
import time
from nltk.corpus import stopwords

# Initialize NLTK data
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("NLTK data downloaded successfully!")
except Exception as e:
    print(f"Warning: NLTK data download failed but continuing anyway. Error: {e}")

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load SpaCy NLP Model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Use a smaller model for faster performance (for now distilgpt2)
MODEL_NAME = "distilgpt2"  # Replace later with a larger model if needed

print("Loading model...")
start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
    device_map="auto"
)

# Create text generation pipeline; using max_length here for generation output control.
resume_analyzer = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=300  # Limit overall output size
)

print(f"Model loaded in {time.time() - start_time:.2f} seconds")

# Function to extract text from PDF/DOCX - process limited pages/paragraphs for speed
def extract_text(file):
    text = ""
    file_path = file.name
    try:
        if file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                # Process all pages for comprehensive analysis
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            return None, "Unsupported file format. Please upload a PDF or DOCX file."
        return text.strip() if text.strip() else None, "No text could be extracted."
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

# Improved section extraction with better regex patterns and structure recognition
def extract_resume_sections(text):
    sections = {}
    
    # Common section headers in resumes
    section_patterns = {
        'summary': r'(profile|summary|objective|about me).*?(?=experience|education|skills|projects|certifications|$)',
        'experience': r'(experience|work|employment|professional background).*?(?=education|skills|projects|certifications|references|$)',
        'education': r'(education|academic|qualification|degree).*?(?=experience|skills|projects|certifications|references|$)',
        'skills': r'(skills|technologies|proficiencies|technical expertise).*?(?=experience|education|projects|certifications|references|$)',
        'projects': r'(projects|personal projects|key projects).*?(?=experience|education|skills|certifications|references|$)',
        'certifications': r'(certifications|licenses|credentials).*?(?=experience|education|skills|projects|references|$)',
        'achievements': r'(achievements|accomplishments|awards).*?(?=experience|education|skills|projects|certifications|references|$)',
        'languages': r'(languages|language proficiency).*?(?=experience|education|skills|projects|certifications|references|$)'
    }
    
    # Extract each section
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections[section_name] = match.group(0)
    
    # Extract contact information and personal details
    contact_info = {}
    # Extract email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if email_match:
        contact_info['email'] = email_match.group(0)
    
    # Extract phone number (various formats)
    phone_match = re.search(r'(\+\d{1,3}[-\.\s]??)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}', text)
    if phone_match:
        contact_info['phone'] = phone_match.group(0)
    
    # Extract LinkedIn (or general URLs)
    linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', text, re.IGNORECASE)
    if linkedin_match:
        contact_info['linkedin'] = linkedin_match.group(0)
    
    sections['contact_info'] = contact_info
    
    # Use SpaCy for named entity recognition to extract additional information
    doc = nlp(text[:5000])  # Process first 5000 chars for efficiency
    
    # Extract key entities
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        if ent.text not in entities[ent.label_]:
            entities[ent.label_].append(ent.text)
    
    sections['entities'] = entities
    
    return sections

# Enhanced achievement recognition in content evaluation
def detect_achievements(text):
    """
    Improved function to detect achievements in resume text, including
    both quantitative and qualitative achievements.
    """
    achievements = []
    achievement_count = 0
    
    # Patterns for quantitative achievements
    quant_patterns = [
        # Percentages
        r'\b(increased|decreased|improved|reduced|grew|achieved|won|earned|raised)\b.*?\b(\d+[\.,]?\d*)\s*(\%|percent)\b',
        # By X percent
        r'\bby\s+(\d+[\.,]?\d*)\s*(\%|percent)\b',
        # Dollar amounts
        r'\$\s*\d+(?:[,.]\d+)*\b',
        # Numbers with units
        r'\b(\d+[\.,]?\d*)\s*(users|customers|clients|employees|people|students|hours|days|months|years|transactions|sales|calls|meetings|projects|programs|applications|websites|systems)\b',
        # Rankings
        r'\b(ranked|rated|voted|recognized|selected|chosen)\b.*?\b(\d+(?:st|nd|rd|th)|\#\d+|number \d+|top \d+)\b',
        # Performance metrics
        r'\b(reached|exceeded|surpassed|beat|outperformed)\b.*?\b(\d+[\.,]?\d*)\b'
    ]
    
    # Patterns for qualitative achievements
    qual_patterns = [
        # Awards and recognitions
        r'\b(received|awarded|earned|won|granted|presented with|recognized with|honored with)\b.*?\b(award|recognition|honor|prize|medal|certificate|scholarship|fellowship|grant)\b',
        # Leadership achievements
        r'\b(led|directed|managed|supervised|oversaw|coordinated|spearheaded|headed)\b.*?\b(team|group|project|initiative|department|organization|company|effort)\b',
        # Project completion
        r'\b(completed|delivered|launched|implemented|deployed|developed|created|established|instituted)\b.*?\b(project|program|system|application|website|platform|solution|initiative)\b',
        # Process improvements
        r'\b(improved|optimized|enhanced|streamlined|simplified|automated|accelerated)\b.*?\b(process|workflow|procedure|operation|system|method|approach)\b',
        # Problem solving
        r'\b(solved|resolved|addressed|fixed|troubleshot|debugged|overcame)\b.*?\b(problem|issue|challenge|bug|error|difficulty|obstacle)\b'
    ]
    
    # Find quantitative achievements
    for pattern in quant_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            achievement_count += len(matches)
            for match in matches[:3]:  # Capture a few example matches
                if isinstance(match, tuple):
                    match_text = ' '.join([m for m in match if m])
                else:
                    match_text = match
                context = find_context(text, match_text, 20, 60)
                if context and len(context) > 20:
                    achievements.append(context.strip())
    
    # Find qualitative achievements
    for pattern in qual_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            achievement_count += len(matches)
            for match in matches[:2]:  # Capture fewer qualitative examples
                if isinstance(match, tuple):
                    match_text = ' '.join([m for m in match if m])
                else:
                    match_text = match
                context = find_context(text, match_text, 20, 60)
                if context and len(context) > 20:
                    achievements.append(context.strip())
    
    # Limit achievements to avoid overwhelming results
    unique_achievements = list(set(achievements))[:5]
    
    return {
        'count': achievement_count,
        'examples': unique_achievements
    }

# Helper function to find context around a match
def find_context(text, match_text, min_chars, max_chars):
    match_pos = text.lower().find(match_text.lower())
    if match_pos == -1:
        return None
    
    # Find the start of the sentence
    start = max(0, match_pos - min_chars)
    while start > 0 and text[start] not in '.!?\n':
        start -= 1
    if start > 0:
        start += 1  # Skip the period
    
    # Find the end of the sentence
    end = min(len(text), match_pos + len(match_text) + min_chars)
    while end < len(text) and text[end] not in '.!?\n':
        end += 1
    if end < len(text):
        end += 1  # Include the period
    
    # If context is too long, trim it
    context = text[start:end].strip()
    if len(context) > max_chars:
        if match_pos - start > end - (match_pos + len(match_text)):
            # If more context before match, trim beginning
            context = "..." + context[-(max_chars-3):]
        else:
            # If more context after match, trim end
            context = context[:max_chars-3] + "..."
    
    return context

# Update comprehensive_resume_evaluation to use improved dynamic job title analysis
def comprehensive_resume_evaluation(text, job_title, resume_sections):
    # Initialize evaluation metrics
    metrics = {
        "overall_score": 5.0,  # Base score (will be adjusted)
        "format_score": 0,
        "content_score": 0,
        "relevance_score": 0,
        "completeness_score": 0,
        "clarity_score": 0,
        "strengths": [],
        "improvements": [],
        "job_specific_feedback": [],
        "achievements": []
    }
    
    # Normalize text for analysis
    text_lower = text.lower()
    job_title_lower = job_title.lower()
    
    # Get job title analysis for dynamic evaluations
    job_analysis = analyze_job_title(job_title)
    
    # ---- 1. FORMAT EVALUATION ----
    # Check resume length
    word_count = len(text.split())
    if word_count < 200:
        metrics["format_score"] -= 1
        metrics["improvements"].append("Resume appears too short (under 200 words). Add more detailed content.")
    elif word_count > 200 and word_count < 600:
        metrics["format_score"] += 0.5
        metrics["strengths"].append("Resume length is appropriate for quick review.")
    elif word_count > 1200:
        metrics["format_score"] -= 0.5
        metrics["improvements"].append("Resume may be too lengthy. Consider condensing to 1-2 pages for better readability.")
    
    # Check for section structure
    section_score = min(3, len(resume_sections)) / 3  # Score based on number of identified sections
    metrics["format_score"] += section_score * 2
    
    if len(resume_sections) >= 3:
        metrics["strengths"].append(f"Good resume structure with {len(resume_sections)} distinct sections identified.")
    else:
        metrics["improvements"].append("Resume could benefit from better section organization and clearer headings.")
    
    # ---- 2. CONTENT EVALUATION ----
    # Check for contact information
    if 'contact_info' in resume_sections and resume_sections['contact_info']:
        metrics["content_score"] += 1
        metrics["strengths"].append("Contact information is present and properly formatted.")
    else:
        metrics["improvements"].append("Ensure contact information is clearly visible at the top of your resume.")
    
    # Check for quantifiable achievements with improved detection
    achievements_data = detect_achievements(text)
    achievement_count = achievements_data['count']
    metrics["achievements"] = achievements_data['examples']  # Store examples for later use
    
    if achievement_count >= 5:
        metrics["content_score"] += 2
        metrics["strengths"].append(f"Strong use of achievements ({achievement_count} identified).")
    elif achievement_count >= 2:
        metrics["content_score"] += 1
        metrics["strengths"].append("Some achievements present. Consider adding more quantifiable results.")
    else:
        metrics["improvements"].append("Add more achievements with specific metrics (%, $, numbers) and results to demonstrate impact.")
    
    # Special handling for students vs. professionals
    is_student = False
    if 'education' in resume_sections:
        is_student = re.search(r'(current|enrolled|attending|studying|expected grad|to graduate|in progress)', 
                              resume_sections['education'], re.IGNORECASE) is not None
        
    if is_student:
        # For students with less work experience, projects and education matter more
        if 'projects' in resume_sections:
            project_length = len(resume_sections['projects'].split())
            if project_length > 150:
                metrics["content_score"] += 1.5
                metrics["strengths"].append("Strong project section demonstrating practical experience.")
            elif project_length > 50:
                metrics["content_score"] += 0.75
                metrics["strengths"].append("Good project section. Consider expanding with more technical details.")
        else:
            metrics["improvements"].append("As a student, adding a detailed projects section will significantly strengthen your resume.")
    
    # Check for action verbs
    action_verbs = [
        'led', 'managed', 'developed', 'created', 'implemented', 'designed', 'coordinated', 
        'analyzed', 'built', 'launched', 'achieved', 'delivered', 'improved', 'increased',
        'decreased', 'reduced', 'optimized', 'transformed', 'streamlined', 'spearheaded',
        'authored', 'negotiated', 'secured', 'initiated', 'established'
    ]
    
    verb_count = sum(1 for verb in action_verbs if re.search(r'\b' + verb + r'\b', text_lower))
    verb_score = min(2, verb_count / 5)  # Max 2 points for action verbs
    metrics["content_score"] += verb_score
    
    if verb_score >= 1.5:
        metrics["strengths"].append("Excellent use of strong action verbs throughout resume.")
    elif verb_score >= 0.5:
        metrics["strengths"].append("Good use of some action verbs. Consider strengthening bullet points with more impactful verbs.")
    else:
        metrics["improvements"].append("Use more strong action verbs to begin your achievement statements.")
    
    # ---- 3. RELEVANCE EVALUATION ----
    # Find skill/keyword matches based on the job title
    industry_keywords = get_industry_keywords(job_title_lower)
    
    matched_keywords = []
    for keyword in industry_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            matched_keywords.append(keyword)
    
    keyword_match_ratio = len(matched_keywords) / max(1, len(industry_keywords))
    relevance_boost = min(3, keyword_match_ratio * 5)  # Max 3 points for relevant keywords
    metrics["relevance_score"] += relevance_boost
    
    if len(matched_keywords) > 5:
        metrics["strengths"].append(f"Strong alignment with {job_title} role (matched keywords: {', '.join(matched_keywords[:5])} and {len(matched_keywords)-5} more).")
    elif len(matched_keywords) > 0:
        metrics["strengths"].append(f"Some alignment with {job_title} role (matched keywords: {', '.join(matched_keywords)}).")
    else:
        metrics["improvements"].append(f"Resume lacks keywords relevant to the {job_title} role. Consider adding industry-specific terms.")
    
    # Get role-specific skills based on job analysis
    if job_analysis['skills']:
        role_specific_skills = job_analysis['skills']
        role_matches = [skill for skill in role_specific_skills if re.search(r'\b' + re.escape(skill) + r'\b', text_lower)]
        
        # Add feedback based on job analysis matches
        if len(role_matches) >= 3:
            metrics["relevance_score"] += 1
            metrics["strengths"].append(f"Good use of {job_title}-specific skills: {', '.join(role_matches[:3])}.")
        elif len(role_matches) > 0:
            metrics["strengths"].append(f"Some {job_title}-specific skills mentioned: {', '.join(role_matches)}. Consider adding more.")
        else:
            # Generate recommendations based on job analysis
            missing_skills = role_specific_skills[:3]
            metrics["improvements"].append(f"Add specific skills relevant to {job_title} positions, such as: {', '.join(missing_skills)}.")
    
    # Check if job title is explicitly mentioned
    if job_title_lower in text_lower or any(term in text_lower for term in job_title_lower.split() if len(term) > 3):
        metrics["relevance_score"] += 1
        metrics["strengths"].append(f"Resume explicitly mentions or targets the {job_title} role.")
    else:
        metrics["improvements"].append(f"Consider explicitly mentioning your interest in or experience with {job_title} roles.")
    
    # Adjust relevance score based on seniority match
    if job_analysis['seniority'] != "mid-level":  # If not default
        seniority_terms = {
            'junior': ['junior', 'entry', 'entry-level', 'associate', 'trainee', 'intern'],
            'senior': ['senior', 'sr', 'lead', 'principal', 'staff', 'expert', 'specialist'],
            'manager': ['manager', 'head', 'director', 'chief', 'vp', 'executive']
        }
        
        # Check if the resume mentions the seniority level
        if any(term in text_lower for term in seniority_terms[job_analysis['seniority']]):
            metrics["relevance_score"] += 0.5
            metrics["strengths"].append(f"Resume appropriately highlights {job_analysis['seniority']} level experience.")
        else:
            metrics["improvements"].append(f"Consider emphasizing your {job_analysis['seniority']} level experience or qualifications.")
    
    # ---- 4. COMPLETENESS EVALUATION ----
    essential_sections = ['experience', 'education', 'skills']
    found_essential = [section for section in essential_sections if section in resume_sections]
    
    completeness_score = len(found_essential) / len(essential_sections) * 3  # Max 3 points
    metrics["completeness_score"] += completeness_score
    
    if 'experience' in resume_sections:
        # Check experience detail
        exp_text = resume_sections['experience']
        exp_length = len(exp_text.split())
        if exp_length > 200:
            metrics["completeness_score"] += 1
            metrics["strengths"].append("Detailed work experience section with comprehensive information.")
        elif exp_length > 100:
            metrics["completeness_score"] += 0.5
            metrics["strengths"].append("Adequate work experience section.")
        else:
            metrics["improvements"].append("Expand your work experience section with more details about responsibilities and achievements.")
    else:
        # Only penalize missing experience section for non-students
        if not is_student:
            metrics["improvements"].append("Add a dedicated work experience section to your resume.")
    
    # Check for chronological gaps (basic implementation)
    if 'experience' in resume_sections:
        exp_text = resume_sections['experience']
        years = re.findall(r'\b(19|20)\d{2}\b', exp_text)
        if len(years) >= 2:
            years = [int(year) for year in years]
            years.sort()
            if max(years) - min(years) > 10 and len(years) < 3:
                metrics["improvements"].append("There may be gaps in your work history. Consider addressing these or using a functional resume format.")
    
    # ---- 5. CLARITY EVALUATION ----
    # Start with a base clarity score
    metrics["clarity_score"] = 3.0  # Start with a moderate score
    
    # Check for concise language
    sentences = nltk.sent_tokenize(text)
    avg_words_per_sentence = word_count / max(1, len(sentences))
    
    if avg_words_per_sentence > 25:
        metrics["clarity_score"] -= 1
        metrics["improvements"].append("Your sentences are quite long (avg. {:.1f} words). Use more concise language for better readability.".format(avg_words_per_sentence))
    elif 10 <= avg_words_per_sentence <= 20:
        metrics["clarity_score"] += 1
        metrics["strengths"].append("Good use of concise language throughout your resume.")
    elif avg_words_per_sentence < 10:
        metrics["clarity_score"] += 0.5
        metrics["strengths"].append("Very concise language used throughout your resume.")
    
    # Check for bullet points and formatting
    bullet_points = len(re.findall(r'[•\-\*]\s', text))
    if bullet_points >= 5:
        metrics["clarity_score"] += 0.5
        metrics["strengths"].append("Good use of bullet points for clear organization.")
    elif bullet_points == 0:
        metrics["clarity_score"] -= 0.5
        metrics["improvements"].append("Consider using bullet points to organize information more clearly.")
    
    # Check for passive voice (basic implementation)
    passive_count = len(re.findall(r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', text_lower))
    if passive_count > 5:
        metrics["clarity_score"] -= 1
        metrics["improvements"].append("Consider replacing passive voice with active voice for stronger impact.")
    elif passive_count <= 2:
        metrics["clarity_score"] += 0.5
        metrics["strengths"].append("Good use of active voice throughout your resume.")
    
    # Check for clear section headings
    section_headings = len([line for line in text.split('\n') if line.strip() and len(line.strip()) < 30 and line.isupper()])
    if section_headings >= 3:
        metrics["clarity_score"] += 0.5
        metrics["strengths"].append("Clear section headings make your resume easy to navigate.")
    
    # Check for consistent formatting
    inconsistent_formatting = len(re.findall(r'[^\s\w\.\,\;\:\-\(\)\&\@\#\$\%\/\'\"\+]', text))
    if inconsistent_formatting > 10:
        metrics["clarity_score"] -= 0.5
        metrics["improvements"].append("Consider simplifying formatting for better clarity and ATS compatibility.")
    
    # Ensure clarity score stays within bounds
    metrics["clarity_score"] = max(0, min(5, metrics["clarity_score"]))
    
    # ---- CALCULATE FINAL SCORE ----
    # Weight each category
    final_score = (
        metrics["format_score"] * 0.15 +
        metrics["content_score"] * 0.30 +
        metrics["relevance_score"] * 0.30 +
        metrics["completeness_score"] * 0.15 +
        metrics["clarity_score"] * 0.10
    )
    
    # Normalize to 1-10 scale
    metrics["overall_score"] = max(1, min(10, final_score + 5))  # Add 5 as base score
    
    # Special adjustment for students
    if is_student:
        # For students, we're more lenient on experience but stricter on education/projects
        if 'education' in resume_sections and 'projects' in resume_sections:
            metrics["overall_score"] = min(10, metrics["overall_score"] + 0.5)
    
    # Cap strengths and improvements
    metrics["strengths"] = metrics["strengths"][:7]  # Limit to top 7 strengths
    metrics["improvements"] = metrics["improvements"][:7]  # Limit to top 7 improvements
    
    return metrics

# Add a dynamic job title analysis function that works for any role
def analyze_job_title(job_title):
    """
    Dynamically analyze any job title to extract:
    1. Job category/industry
    2. Likely required skills
    3. Special keywords relevant to the role
    
    This works for any job title without relying solely on hardcoded values.
    """
    job_title_lower = job_title.lower()
    words = job_title_lower.split()
    
    # Common job categories and their associated keywords
    job_categories = {
        'engineering': ['engineer', 'engineering', 'developer', 'programmer', 'architect', 'coder', 'technician', 'specialist'],
        'management': ['manager', 'director', 'supervisor', 'lead', 'chief', 'head', 'coordinator', 'executive'],
        'analysis': ['analyst', 'researcher', 'scientist', 'specialist', 'consultant', 'data scientist', 'business analyst'],
        'design': ['designer', 'architect', 'ui/ux', 'ux', 'ui', 'graphic', 'product designer', 'visual designer'],
        'support': ['support', 'assistant', 'help', 'technician', 'service', 'customer support', 'technical support'],
        'sales': ['sales', 'account', 'representative', 'business development', 'client', 'sales manager', 'account executive'],
        'marketing': ['marketing', 'brand', 'seo', 'content', 'social media', 'growth', 'digital marketing', 'communications'],
        'finance': ['finance', 'accounting', 'financial', 'accountant', 'bookkeeper', 'controller', 'treasurer', 'auditor'],
        'hr': ['hr', 'human resources', 'recruiter', 'talent', 'people', 'recruitment', 'personnel', 'benefits'],
        'healthcare': ['doctor', 'nurse', 'physician', 'medical', 'clinical', 'health', 'therapist', 'pharmacist', 'dentist'],
        'education': ['teacher', 'professor', 'instructor', 'educator', 'tutor', 'trainer', 'counselor', 'principal'],
        'legal': ['lawyer', 'attorney', 'legal', 'counsel', 'paralegal', 'compliance', 'regulatory', 'contracts'],
        'creative': ['writer', 'editor', 'artist', 'creative', 'content', 'producer', 'copywriter', 'journalist'],
        'customer': ['customer', 'client', 'service', 'success', 'support', 'experience', 'relationship', 'account manager'],
        'administrative': ['admin', 'administrative', 'coordinator', 'secretary', 'clerk', 'receptionist', 'office manager'],
        'operations': ['operations', 'logistic', 'supply chain', 'warehouse', 'inventory', 'production', 'facility'],
        # New categories
        'research': ['researcher', 'research assistant', 'research associate', 'lab technician', 'scientist', 'investigator'],
        'it': ['it', 'information technology', 'systems', 'network', 'infrastructure', 'support', 'administrator'],
        'data': ['data', 'analytics', 'statistics', 'metrics', 'reporting', 'business intelligence', 'data science'],
        'project': ['project manager', 'program manager', 'scrum master', 'project coordinator', 'delivery manager'],
        'consulting': ['consultant', 'advisor', 'strategist', 'specialist', 'expert', 'coach', 'mentor'],
        'manufacturing': ['manufacturing', 'production', 'assembly', 'quality control', 'plant', 'industrial'],
        'construction': ['construction', 'builder', 'contractor', 'architect', 'site manager', 'project manager'],
        'hospitality': ['hospitality', 'hotel', 'restaurant', 'chef', 'catering', 'food service', 'tourism'],
        'retail': ['retail', 'store', 'merchandising', 'buyer', 'shop', 'sales floor', 'inventory'],
        'media': ['media', 'journalist', 'reporter', 'broadcaster', 'producer', 'editor', 'content creator'],
        'real_estate': ['real estate', 'property', 'broker', 'agent', 'leasing', 'property manager'],
        'nonprofit': ['nonprofit', 'ngo', 'charity', 'foundation', 'fundraising', 'program coordinator'],
        'government': ['government', 'public sector', 'civil service', 'policy', 'administration', 'public affairs'],
        'science': ['scientist', 'researcher', 'lab', 'research', 'development', 'r&d', 'laboratory'],
        'environmental': ['environmental', 'sustainability', 'conservation', 'ecology', 'climate', 'energy'],
        'security': ['security', 'cybersecurity', 'information security', 'security analyst', 'risk', 'compliance'],
        'transportation': ['transportation', 'logistics', 'shipping', 'fleet', 'driver', 'dispatcher'],
        'agriculture': ['agriculture', 'farming', 'agribusiness', 'horticulture', 'crop', 'livestock'],
        'fitness': ['fitness', 'personal trainer', 'coach', 'instructor', 'wellness', 'health'],
        'insurance': ['insurance', 'underwriter', 'claims', 'actuary', 'risk assessment', 'benefits'],
        'telecommunications': ['telecommunications', 'network', 'communications', 'telecom', 'wireless'],
        'automotive': ['automotive', 'mechanic', 'technician', 'service', 'repair', 'dealership'],
        'aviation': ['aviation', 'pilot', 'aircraft', 'airline', 'airport', 'flight'],
        'pharmaceutical': ['pharmaceutical', 'pharma', 'drug development', 'clinical research', 'biotech'],
        'entertainment': ['entertainment', 'film', 'music', 'gaming', 'production', 'artist'],
        'fashion': ['fashion', 'apparel', 'designer', 'merchandiser', 'buyer', 'stylist'],
        'social_services': ['social services', 'counselor', 'social worker', 'case manager', 'community'],
        'quality': ['quality assurance', 'qa', 'quality control', 'testing', 'compliance', 'inspector'],
        'maintenance': ['maintenance', 'facilities', 'repair', 'technician', 'service', 'building'],
        'energy': ['energy', 'power', 'utilities', 'renewable', 'electrical', 'oil and gas']
    }
    
    # Domain-specific dictionaries for popular fields
    domain_keywords = {
        'software': ['programming', 'coding', 'development', 'algorithms', 'testing', 'debugging', 
                    'software architecture', 'apis', 'web', 'mobile', 'frontend', 'backend', 'full stack',
                    'object-oriented', 'functional programming', 'version control', 'ci/cd', 'unit testing',
                    'integration testing', 'code review', 'agile development', 'scrum', 'devops'],
        
        'data': ['analytics', 'big data', 'data mining', 'statistics', 'data visualization', 
                'machine learning', 'artificial intelligence', 'deep learning', 'predictive modeling',
                'data warehousing', 'etl', 'bi', 'reporting', 'dashboards', 'data science',
                'data engineering', 'data architecture', 'data governance', 'data quality'],
        
        'cloud': ['aws', 'azure', 'gcp', 'infrastructure', 'virtualization', 'containers',
                 'microservices', 'serverless', 'iaas', 'paas', 'saas', 'cloud security',
                 'cloud architecture', 'devops', 'automation', 'scalability', 'cloud migration',
                 'hybrid cloud', 'multi-cloud', 'cloud native', 'cloud optimization'],
        
        'security': ['cybersecurity', 'infosec', 'security protocols', 'penetration testing', 
                    'vulnerability assessment', 'encryption', 'authentication', 'compliance',
                    'security audits', 'threat detection', 'incident response', 'security architecture',
                    'network security', 'application security', 'cloud security', 'devsecops'],
        
        'project': ['project management', 'agile', 'scrum', 'kanban', 'waterfall', 'prince2',
                   'risk management', 'requirements gathering', 'stakeholder management', 
                   'scheduling', 'budget management', 'resource allocation', 'project planning',
                   'project execution', 'project monitoring', 'project closure'],
        
        'finance': ['financial analysis', 'accounting', 'budgeting', 'forecasting', 'financial reporting',
                   'financial modeling', 'tax', 'audit', 'compliance', 'risk assessment',
                   'investment analysis', 'portfolio management', 'financial planning',
                   'cost analysis', 'revenue management', 'financial strategy'],
        
        'marketing': ['market research', 'branding', 'advertising', 'digital marketing', 'seo', 'sem', 
                     'social media marketing', 'content marketing', 'email marketing', 'campaign management',
                     'marketing analytics', 'conversion optimization', 'marketing automation',
                     'customer acquisition', 'marketing strategy', 'brand management'],
        
        'healthcare': ['patient care', 'clinical procedures', 'medical terminology', 'healthcare regulations',
                      'electronic health records', 'patient management', 'medical coding',
                      'treatment planning', 'disease management', 'healthcare compliance',
                      'medical billing', 'healthcare technology', 'patient safety'],
        
        'hr': ['recruitment', 'talent acquisition', 'onboarding', 'employee relations', 'performance management',
              'compensation', 'benefits', 'hr policies', 'organizational development',
              'training', 'employee engagement', 'diversity and inclusion', 'hr analytics',
              'succession planning', 'workforce planning', 'labor relations'],
        
        'legal': ['legal research', 'contracts', 'compliance', 'litigation', 'negotiation',
                 'legal documentation', 'case management', 'legal advice', 'legal analysis',
                 'regulatory affairs', 'intellectual property', 'corporate law', 'employment law',
                 'legal technology', 'legal writing', 'legal ethics'],

        'education': ['curriculum development', 'instructional design', 'educational technology',
                     'student assessment', 'classroom management', 'special education',
                     'online learning', 'educational leadership', 'student engagement',
                     'educational psychology', 'teaching methods', 'education policy'],

        'manufacturing': ['production planning', 'quality control', 'lean manufacturing',
                         'supply chain', 'inventory management', 'process improvement',
                         'industrial engineering', 'production scheduling', 'manufacturing operations',
                         'equipment maintenance', 'safety compliance', 'cost reduction'],

        'retail': ['merchandising', 'inventory management', 'retail operations',
                  'store management', 'visual merchandising', 'pos systems',
                  'customer service', 'sales management', 'retail analytics',
                  'loss prevention', 'retail marketing', 'e-commerce'],

        'construction': ['project planning', 'site management', 'safety compliance',
                        'building codes', 'construction management', 'cost estimation',
                        'contract administration', 'quality control', 'scheduling',
                        'resource management', 'risk assessment'],

        'hospitality': ['guest services', 'hotel operations', 'food service',
                       'event planning', 'revenue management', 'hospitality marketing',
                       'customer experience', 'reservation systems', 'facility management',
                       'hospitality analytics', 'tourism management'],

        'media': ['content creation', 'media production', 'broadcasting',
                 'digital media', 'social media', 'content strategy',
                 'media planning', 'audience engagement', 'media analytics',
                 'multimedia production', 'content management'],

        'automotive': ['vehicle maintenance', 'automotive technology', 'diagnostics',
                      'repair procedures', 'quality control', 'service management',
                      'parts inventory', 'automotive systems', 'customer service',
                      'technical documentation'],

        'environmental': ['environmental compliance', 'sustainability', 'environmental impact',
                         'waste management', 'environmental monitoring', 'conservation',
                         'renewable energy', 'environmental policy', 'climate change',
                         'environmental assessment'],

        'telecommunications': ['network infrastructure', 'telecom systems', 'wireless technology',
                             'network maintenance', 'telecom operations', 'service delivery',
                             'network security', 'telecommunications policy', 'customer support',
                             'telecommunications regulations'],

        'pharmaceutical': ['drug development', 'clinical research', 'regulatory compliance',
                         'quality control', 'pharmaceutical manufacturing', 'research and development',
                         'drug safety', 'clinical trials', 'pharmaceutical marketing',
                         'pharmaceutical regulations'],

        'nonprofit': ['program management', 'fundraising', 'grant writing',
                     'community outreach', 'volunteer management', 'nonprofit operations',
                     'donor relations', 'impact assessment', 'advocacy',
                     'nonprofit compliance'],

        'government': ['public administration', 'policy implementation', 'regulatory compliance',
                      'public service', 'government operations', 'public policy',
                      'legislative affairs', 'government relations', 'public programs',
                      'civic engagement'],

        'real_estate': ['property management', 'real estate transactions', 'leasing',
                       'property valuation', 'real estate marketing', 'market analysis',
                       'property maintenance', 'tenant relations', 'real estate finance',
                       'real estate development'],

        'insurance': ['risk assessment', 'underwriting', 'claims processing',
                     'insurance policies', 'actuarial analysis', 'insurance regulations',
                     'policy administration', 'insurance marketing', 'customer service',
                     'insurance compliance'],

        'transportation': ['logistics management', 'fleet operations', 'transportation planning',
                         'route optimization', 'shipping coordination', 'compliance',
                         'safety management', 'transportation regulations', 'cost control',
                         'fleet maintenance']
    }
    
    # Identify job category
    matched_categories = []
    for category, keywords in job_categories.items():
        if any(keyword in job_title_lower for keyword in keywords):
            matched_categories.append(category)
    
    # Identify domain
    matched_domains = []
    for domain, _ in domain_keywords.items():
        if domain in job_title_lower:
            matched_domains.append(domain)
    
    # Generate base skills required for this job title
    skills = set()
    
    # Add skills based on job category
    for category in matched_categories:
        # Generic skills by job category
        if category == 'engineering':
            skills.update(['problem solving', 'technical knowledge', 'debugging', 'testing', 'documentation'])
        elif category == 'management':
            skills.update(['leadership', 'team management', 'strategy', 'planning', 'budget management', 'decision making'])
        elif category == 'analysis':
            skills.update(['data analysis', 'critical thinking', 'research', 'reporting', 'attention to detail'])
        elif category == 'design':
            skills.update(['creativity', 'visual communication', 'design principles', 'user experience', 'creative tools'])
        # Add more categories as needed
    
    # Add domain-specific skills
    for domain in matched_domains:
        if domain in domain_keywords:
            # Add a subset of domain keywords to avoid overwhelming results
            domain_skills = domain_keywords[domain]
            skills.update(domain_skills[:10])  # Add up to 10 domain-specific skills
    
    # Extract any specific technologies or tools mentioned in the job title
    tech_patterns = [
        r'(python|java|javascript|react|angular|node\.js|aws|azure|gcp|sql|nosql|sap|salesforce)',
        r'(excel|word|powerpoint|tableau|power bi|photoshop|illustrator|figma)',
        r'(kubernetes|docker|jenkins|git|terraform|ansible|puppet|chef)'
    ]
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, job_title_lower)
        skills.update(matches)
    
    # Associate seniority level with expectations
    seniority = "mid-level"  # Default
    seniority_terms = {
        'junior': ['junior', 'entry', 'entry-level', 'associate', 'trainee', 'intern'],
        'mid-level': ['mid', 'regular', 'experienced', 'intermediate'],
        'senior': ['senior', 'sr', 'lead', 'principal', 'staff', 'expert', 'specialist'],
        'manager': ['manager', 'head', 'director', 'chief', 'vp', 'executive']
    }
    
    for level, terms in seniority_terms.items():
        if any(term in job_title_lower for term in terms):
            seniority = level
            break
    
    # Handle special composite job titles
    if 'full stack' in job_title_lower or 'fullstack' in job_title_lower:
        skills.update(['frontend development', 'backend development', 'databases', 'api integration'])
    
    if 'devops' in job_title_lower:
        skills.update(['ci/cd', 'infrastructure automation', 'containerization', 'monitoring', 'cloud platforms'])
    
    # Generate unique result set
    return {
        'categories': matched_categories,
        'domains': matched_domains,
        'skills': list(skills),
        'seniority': seniority
    }

# Update the get_industry_keywords function to be more dynamic
def get_industry_keywords(job_title):
    """
    Get relevant keywords for any job title using a combination of:
    1. Pre-defined industry keywords
    2. Dynamic job title analysis
    3. Common professional skills
    """
    # Base keywords that apply to most jobs
    general_keywords = [
        'leadership', 'teamwork', 'communication', 'project management', 
        'problem solving', 'analytical', 'detail-oriented', 'deadline',
        'collaboration', 'innovation', 'initiative', 'results-driven'
    ]
    
    # Industry and role-specific keywords from predefined lists
    industry_keywords = {
        # Tech/IT roles
        'software': ['python', 'java', 'javascript', 'react', 'angular', 'node.js', 'api', 
                    'aws', 'azure', 'cloud', 'devops', 'ci/cd', 'git', 'agile', 'scrum'],
        'data': ['sql', 'python', 'r', 'machine learning', 'ai', 'tableau', 'power bi', 
                'statistics', 'analytics', 'big data', 'data mining', 'visualization'],
        'engineer': ['design', 'development', 'testing', 'debugging', 'optimization', 
                   'architecture', 'algorithms', 'documentation', 'requirements'],
        'devops': ['cloud', 'aws', 'azure', 'gcp', 'kubernetes', 'docker', 'jenkins', 
                 'terraform', 'ansible', 'ci/cd', 'automation', 'monitoring'],
        'security': ['cybersecurity', 'penetration testing', 'vulnerability', 'compliance', 
                   'risk assessment', 'security protocols', 'encryption', 'firewall'],
        'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'ec2', 's3', 'lambda', 'serverless',
                 'iaas', 'paas', 'saas', 'virtualization', 'containerization', 'kubernetes',
                 'docker', 'terraform', 'cloudformation', 'ansible', 'puppet', 'chef',
                 'infrastructure as code', 'load balancing', 'auto scaling', 'cloudwatch',
                 'route 53', 'vpc', 'eks', 'ecs', 'security groups', 'cloud security',
                 'elasticache', 'rds', 'dynamodb', 'nosql', 'microservices', 'api gateway',
                 'cloud migration', 'hybrid cloud', 'multi-cloud', 'disaster recovery',
                 'high availability', 'fault tolerance', 'cloudfront', 'cloud storage',
                 'blob storage', 'cloud networking', 'cloud monitoring', 'devops', 'sre',
                 'site reliability', 'continuous integration', 'continuous deployment',
                 'ci/cd pipeline', 'git', 'github', 'gitlab', 'bitbucket', 'jenkins',
                 'cloud cost optimization', 'cloud architecture', 'cloud solutions'],
        
        # Business roles
        'manager': ['leadership', 'strategy', 'team building', 'budgeting', 'forecasting', 
                  'performance management', 'kpi', 'process improvement'],
        'marketing': ['brand', 'campaigns', 'digital marketing', 'seo', 'social media', 
                    'content strategy', 'analytics', 'market research', 'customer acquisition'],
        'sales': ['revenue growth', 'client acquisition', 'negotiation', 'crm', 'pipeline', 
                'closing deals', 'relationship management', 'quota'],
        'finance': ['financial analysis', 'forecasting', 'budget', 'profit', 'loss', 
                  'accounting', 'reporting', 'compliance', 'audit'],
        'hr': ['recruitment', 'talent acquisition', 'employee relations', 'performance management', 
             'benefits', 'compensation', 'onboarding', 'hr policies'],
        
        # Healthcare roles
        'nurse': ['patient care', 'clinical', 'healthcare', 'treatment', 'medical', 
                'assessment', 'documentation', 'regulations'],
        'doctor': ['diagnosis', 'treatment', 'patient care', 'medical', 'clinical', 
                 'healthcare', 'examination'],
        
        # Legal roles
        'legal': ['contracts', 'compliance', 'regulations', 'litigation', 'legal research', 
                'analysis', 'negotiation', 'documentation'],
        
        # Education roles
        'teacher': ['curriculum', 'instruction', 'assessment', 'classroom management', 
                  'lesson planning', 'student engagement', 'education'],
        
        # Design roles
        'design': ['ux', 'ui', 'user experience', 'wireframes', 'prototypes', 'usability', 
                 'adobe', 'figma', 'sketch', 'creative', 'portfolio']
    }
    
    # Dynamic analysis of the job title
    job_analysis = analyze_job_title(job_title)
    
    # Begin with general keywords
    keywords = general_keywords.copy()
    
    # Add dynamic keywords from job analysis
    if job_analysis['skills']:
        keywords.extend(job_analysis['skills'])
    
    # Add predefined industry keywords when they match
    job_title_lower = job_title.lower()
    for industry, industry_specific_keywords in industry_keywords.items():
        if industry in job_title_lower:
            keywords.extend(industry_specific_keywords)
    
    # Special handling for common job areas not explicitly named
    if any(term in job_title_lower for term in ['full stack', 'fullstack', 'web developer']):
        keywords.extend(['html', 'css', 'javascript', 'frontend', 'backend', 'api', 'responsive design'])
    
    if any(term in job_title_lower for term in ['data scientist', 'data analyst', 'machine learning']):
        keywords.extend(['python', 'r', 'statistics', 'machine learning', 'data visualization', 'sql'])
    
    # Additional keywords based on seniority
    if job_analysis['seniority'] == 'senior' or job_analysis['seniority'] == 'manager':
        keywords.extend(['leadership', 'mentoring', 'strategic planning', 'architecture', 'stakeholder management'])
    
    if job_analysis['seniority'] == 'junior':
        keywords.extend(['eager to learn', 'adaptability', 'attention to detail', 'following instructions'])
    
    # Return unique keywords
    return list(set(keywords))  # Return unique keywords

# Update the job recommendations to be more dynamic as well
def get_advanced_job_recommendations(job_title, industry=None):
    """
    Provide advanced, personalized recommendations for any job title
    by combining predefined recommendations with dynamic job analysis.
    """
    # Dynamic analysis of job title
    job_analysis = analyze_job_title(job_title)
    
    # Common recommendations that apply to most jobs
    common_recommendations = [
        "Tailor your resume keywords to match the specific job description",
        "Include a strong summary statement focused on your value proposition",
        "Quantify achievements with specific metrics (%, $, numbers)",
        "Remove outdated or irrelevant experience",
        "Ensure consistent formatting and proper ATS optimization"
    ]
    
    # Format-related recommendations
    format_recommendations = [
        "Use bullet points for better readability and scanning",
        "Keep your resume to 1-2 pages maximum",
        "Use a clean, professional design with consistent formatting",
        "Place your most relevant experience and skills at the top",
        "Use white space effectively to guide the reader's eye"
    ]
    
    # Pre-defined industry-specific recommendations
    industry_recommendations = {
        # Tech roles
        'software engineer': [
            "Include a GitHub profile or portfolio link prominently",
            "List specific programming languages with proficiency levels",
            "Highlight system design and architecture experience",
            "Showcase contributions to open-source projects",
            "Describe projects with tech stack details and your specific role"
        ],
        'data scientist': [
            "Include specific ML/AI frameworks and tools you've used",
            "Describe projects with specific metrics of improvement",
            "Mention data sizes you've worked with (if impressive)",
            "Highlight any published research or technical blog posts",
            "Showcase business impact of your data work, not just technical details"
        ],
        'product manager': [
            "Quantify product outcomes and business impact",
            "Describe cross-functional collaboration experience",
            "Highlight product lifecycle management experience",
            "Mention specific methodologies (Agile, Scrum, etc.)",
            "Include examples of customer-focused solutions"
        ],
        'cloud engineer': [
            "Highlight specific cloud platforms you've worked with (AWS, Azure, GCP)",
            "Detail your experience with infrastructure as code (Terraform, CloudFormation)",
            "Showcase cloud architecture designs and optimizations you've implemented",
            "Include security best practices and implementations in cloud environments",
            "Mention experience with containers (Docker) and orchestration (Kubernetes)"
        ],
        'designer': [
            "Include a link to your portfolio showcasing your best work",
            "Highlight specific design tools and software proficiency",
            "Mention design methodologies and processes you follow",
            "Showcase measurable results from your design projects",
            "Include examples of collaboration with developers or other teams"
        ],
        'marketing specialist': [
            "Showcase campaigns with measurable ROI or conversion metrics",
            "Highlight experience with specific marketing tools and platforms",
            "Include examples of content creation and audience engagement",
            "Mention experience with analytics and data-driven decisions",
            "Showcase understanding of current marketing trends and channels"
        ],
        'project manager': [
            "Highlight specific project management methodologies (Agile, Scrum, etc.)",
            "Quantify project budgets, timelines, and team sizes managed",
            "Showcase experience with project management tools",
            "Highlight stakeholder management and communication skills",
            "Include examples of successful project deliveries and outcomes"
        ],
        'sales representative': [
            "Quantify sales achievements, quotas, and revenue generated",
            "Highlight experience with CRM systems and sales tools",
            "Showcase successful client relationship management",
            "Include examples of overcoming sales challenges",
            "Mention negotiation strategies and closing techniques"
        ],
        'teacher': [
            "Highlight specific teaching methodologies and approaches",
            "Mention curriculum development experience",
            "Showcase classroom management techniques",
            "Include examples of student achievement improvements",
            "Highlight use of educational technology and tools"
        ],
        'nurse': [
            "List all certifications prominently with expiration dates",
            "Include specific patient care metrics or improvements",
            "Mention experience with specific medical technologies/systems",
            "Highlight emergency response or specialized care experience",
            "Include continuing education and specialized training"
        ]
    }
    
    # Generate recommendations based on job analysis
    dynamic_recommendations = []
    
    # Add category-specific recommendations
    if 'engineering' in job_analysis['categories']:
        dynamic_recommendations.extend([
            "Showcase technical projects with measurable outcomes",
            "Highlight problem-solving skills with specific examples",
            "Include relevant technical certifications and training"
        ])
    
    if 'management' in job_analysis['categories']:
        dynamic_recommendations.extend([
            "Highlight team leadership and development achievements",
            "Showcase budget management and resource allocation experience",
            "Include examples of strategic planning and execution"
        ])
    
    if 'analysis' in job_analysis['categories']:
        dynamic_recommendations.extend([
            "Showcase data analysis skills with specific tools and methods",
            "Highlight projects where your analysis led to important decisions",
            "Include examples of complex problems you've solved through analysis"
        ])
    
    # Add domain-specific recommendations
    if 'software' in job_analysis['domains']:
        dynamic_recommendations.extend([
            "List programming languages and frameworks with proficiency levels",
            "Include a link to your GitHub or portfolio",
            "Highlight software development methodologies you're familiar with"
        ])
    
    if 'data' in job_analysis['domains']:
        dynamic_recommendations.extend([
            "Showcase experience with specific data tools and technologies",
            "Highlight projects involving large datasets or complex analysis",
            "Include examples of insights derived from your data analysis"
        ])
    
    # Add seniority-specific recommendations
    if job_analysis['seniority'] == 'senior':
        dynamic_recommendations.extend([
            "Emphasize leadership and mentoring experience",
            "Highlight strategic contributions to your organization",
            "Showcase deep expertise in your field with specific examples"
        ])
    
    if job_analysis['seniority'] == 'junior':
        dynamic_recommendations.extend([
            "Highlight relevant coursework, projects, and internships",
            "Emphasize eagerness to learn and adaptability",
            "Showcase relevant skills gained from academic or personal projects"
        ])
    
    # Try to find exact job title match in predefined recommendations
    job_title_lower = job_title.lower()
    final_recommendations = []
    
    # Try exact match first
    for key, recommendations in industry_recommendations.items():
        if key in job_title_lower:
            final_recommendations.extend(recommendations)
            break
    
    # If no exact match, build custom recommendations
    if not final_recommendations:
        # Start with common recommendations
        final_recommendations.extend(common_recommendations[:2])
        
        # Add dynamic recommendations based on job analysis
        final_recommendations.extend(dynamic_recommendations[:3])
        
        # Add some format recommendations
        final_recommendations.extend(format_recommendations[:2])
    
    # Ensure we have at least 5 recommendations
    if len(final_recommendations) < 5:
        remaining_needed = 5 - len(final_recommendations)
        final_recommendations.extend(common_recommendations[:remaining_needed])
    
    # Return the top 7 recommendations
    return final_recommendations[:7]

# Use the language model to generate role-specific feedback
def generate_role_specific_feedback(job_title, resume_sections, evaluation_metrics):
    """
    Generate targeted feedback specific to the role using dynamic job analysis 
    and resume evaluation results.
    """
    # Get job analysis for targeted feedback
    job_analysis = analyze_job_title(job_title)
    
    # Start with a default feedback template based on job analysis
    feedback = []
    
    # Add job category-specific feedback
    if job_analysis['categories']:
        main_category = job_analysis['categories'][0] if job_analysis['categories'] else None
        
        if main_category == 'engineering':
            if evaluation_metrics['relevance_score'] < 3:
                feedback.append(f"Your resume needs more technical details relevant to {job_title} positions.")
            else:
                feedback.append(f"Your technical skills align well with {job_title} requirements.")
                
        elif main_category == 'management':
            if 'leadership' in resume_sections.get('experience', '').lower():
                feedback.append(f"Your leadership experience is valuable for a {job_title} role.")
            else:
                feedback.append(f"Highlight your leadership experience and team management skills for {job_title} positions.")
                
        elif main_category == 'design':
            feedback.append("Include a link to your portfolio showcasing relevant design work.")
                
        elif main_category == 'sales':
            if not any(term in resume_sections.get('experience', '').lower() for term in ['revenue', 'quota', 'sales', 'client']):
                feedback.append("Quantify your sales achievements with specific revenue figures or percentages.")
    
        elif main_category == 'healthcare':
            if not any(term in resume_sections.get('experience', '').lower() for term in ['patient', 'clinical', 'medical', 'health']):
                feedback.append("Include specific medical procedures, patient care metrics, and clinical experience details.")
            else:
                feedback.append("Your clinical experience is well-highlighted. Consider adding more quantifiable patient care outcomes.")

        elif main_category == 'finance':
            if not any(term in resume_sections.get('experience', '').lower() for term in ['financial', 'budget', 'revenue', 'analysis']):
                feedback.append("Add specific financial metrics, portfolio performance, or cost-saving achievements.")
            else:
                feedback.append("Your financial expertise is evident. Consider highlighting regulatory compliance experience.")

        elif main_category == 'education':
            if not any(term in resume_sections.get('experience', '').lower() for term in ['teach', 'curriculum', 'student', 'education']):
                feedback.append("Include teaching methodologies, student achievement metrics, and curriculum development experience.")
            else:
                feedback.append("Your teaching experience is clear. Consider adding specific student success metrics.")

        elif main_category == 'technology':
            tech_terms = ['software', 'development', 'programming', 'code', 'technical']
            if not any(term in resume_sections.get('experience', '').lower() for term in tech_terms):
                feedback.append("Add more technical projects, programming languages, and development methodologies.")
            else:
                feedback.append("Your technical background is solid. Consider highlighting system architecture or scalability experience.")

        elif main_category == 'research':
            research_terms = ['research', 'study', 'analysis', 'methodology', 'findings']
            if not any(term in resume_sections.get('experience', '').lower() for term in research_terms):
                feedback.append("Include research methodologies, published works, and significant findings in your field.")
            else:
                feedback.append("Your research experience is evident. Consider highlighting impact and citations of your work.")

        elif main_category == 'marketing':
            marketing_terms = ['campaign', 'marketing', 'brand', 'social media', 'content']
            if not any(term in resume_sections.get('experience', '').lower() for term in marketing_terms):
                feedback.append("Add specific marketing campaigns, ROI metrics, and audience growth achievements.")
            else:
                feedback.append("Your marketing experience is clear. Consider adding more conversion and engagement metrics.")

        elif main_category == 'customer_service':
            service_terms = ['customer', 'service', 'support', 'client', 'resolution']
            if not any(term in resume_sections.get('experience', '').lower() for term in service_terms):
                feedback.append("Include customer satisfaction metrics, resolution rates, and service improvement initiatives.")
            else:
                feedback.append("Your customer service background is evident. Add more specific performance metrics.")

        elif main_category == 'operations':
            ops_terms = ['operations', 'process', 'efficiency', 'optimization', 'workflow']
            if not any(term in resume_sections.get('experience', '').lower() for term in ops_terms):
                feedback.append("Add operational efficiency metrics, process improvements, and resource optimization examples.")
            else:
                feedback.append("Your operations experience is clear. Consider highlighting cost reduction achievements.")

        elif main_category == 'legal':
            legal_terms = ['legal', 'law', 'compliance', 'regulation', 'counsel']
            if not any(term in resume_sections.get('experience', '').lower() for term in legal_terms):
                feedback.append("Include specific areas of law, case outcomes, and regulatory compliance experience.")
            else:
                feedback.append("Your legal background is evident. Consider highlighting notable case victories or settlements.")

        elif main_category == 'creative':
            creative_terms = ['design', 'creative', 'art', 'portfolio', 'visual']
            if not any(term in resume_sections.get('experience', '').lower() for term in creative_terms):
                feedback.append("Include a portfolio link and highlight specific creative projects and their impact.")
            else:
                feedback.append("Your creative work is well-represented. Consider adding more metrics on project outcomes.")

        elif main_category == 'hr':
            hr_terms = ['recruitment', 'hr', 'talent', 'employee', 'personnel']
            if not any(term in resume_sections.get('experience', '').lower() for term in hr_terms):
                feedback.append("Add recruitment metrics, employee engagement initiatives, and HR policy developments.")
            else:
                feedback.append("Your HR experience is clear. Consider highlighting retention and satisfaction metrics.")

        elif main_category == 'manufacturing':
            mfg_terms = ['manufacturing', 'production', 'quality', 'assembly', 'operations']
            if not any(term in resume_sections.get('experience', '').lower() for term in mfg_terms):
                feedback.append("Include manufacturing processes, quality metrics, and production optimization achievements.")
            else:
                feedback.append("Your manufacturing experience is evident. Consider adding more efficiency and quality metrics.")

        elif main_category == 'construction':
            const_terms = ['construction', 'building', 'project', 'site', 'safety']
            if not any(term in resume_sections.get('experience', '').lower() for term in const_terms):
                feedback.append("Add construction project details, safety compliance, and budget management experience.")
            else:
                feedback.append("Your construction experience is clear. Consider highlighting project completion metrics.")

        elif main_category == 'hospitality':
            hosp_terms = ['hospitality', 'guest', 'hotel', 'restaurant', 'service']
            if not any(term in resume_sections.get('experience', '').lower() for term in hosp_terms):
                feedback.append("Include guest satisfaction metrics, revenue management, and service quality achievements.")
            else:
                feedback.append("Your hospitality experience is evident. Add more customer satisfaction metrics.")

        elif main_category == 'retail':
            retail_terms = ['retail', 'sales', 'store', 'merchandise', 'inventory']
            if not any(term in resume_sections.get('experience', '').lower() for term in retail_terms):
                feedback.append("Add retail sales metrics, inventory management, and customer service achievements.")
            else:
                feedback.append("Your retail experience is clear. Consider highlighting sales and efficiency metrics.")

        elif main_category == 'media':
            media_terms = ['media', 'content', 'production', 'broadcast', 'journalism']
            if not any(term in resume_sections.get('experience', '').lower() for term in media_terms):
                feedback.append("Include media production details, audience metrics, and content performance statistics.")
            else:
                feedback.append("Your media experience is evident. Add more audience engagement metrics.")

        elif main_category == 'environmental':
            env_terms = ['environmental', 'sustainability', 'conservation', 'compliance']
            if not any(term in resume_sections.get('experience', '').lower() for term in env_terms):
                feedback.append("Add environmental impact assessments, sustainability initiatives, and compliance achievements.")
            else:
                feedback.append("Your environmental experience is clear. Consider highlighting project outcomes.")

        elif main_category == 'pharmaceutical':
            pharma_terms = ['pharmaceutical', 'clinical', 'research', 'drug', 'development']
            if not any(term in resume_sections.get('experience', '').lower() for term in pharma_terms):
                feedback.append("Include clinical research experience, drug development projects, and regulatory compliance.")
            else:
                feedback.append("Your pharmaceutical experience is evident. Add more research outcomes.")

        elif main_category == 'telecommunications':
            telecom_terms = ['telecommunications', 'network', 'infrastructure', 'wireless']
            if not any(term in resume_sections.get('experience', '').lower() for term in telecom_terms):
                feedback.append("Add network infrastructure projects, service delivery metrics, and technical achievements.")
            else:
                feedback.append("Your telecommunications experience is clear. Highlight network performance metrics.")

        elif main_category == 'automotive':
            auto_terms = ['automotive', 'vehicle', 'repair', 'maintenance', 'service']
            if not any(term in resume_sections.get('experience', '').lower() for term in auto_terms):
                feedback.append("Include automotive repair experience, service metrics, and technical certifications.")
            else:
                feedback.append("Your automotive experience is evident. Add more service quality metrics.")

        elif main_category == 'insurance':
            insurance_terms = ['insurance', 'claims', 'underwriting', 'risk', 'policy']
            if not any(term in resume_sections.get('experience', '').lower() for term in insurance_terms):
                feedback.append("Add insurance policy management, claims processing, and risk assessment experience.")
            else:
                feedback.append("Your insurance experience is clear. Consider highlighting performance metrics.")

        elif main_category == 'real_estate':
            re_terms = ['real estate', 'property', 'leasing', 'sales', 'management']
            if not any(term in resume_sections.get('experience', '').lower() for term in re_terms):
                feedback.append("Include property management experience, sales/leasing metrics, and market analysis.")
            else:
                feedback.append("Your real estate experience is evident. Add more transaction metrics.")

        elif main_category == 'nonprofit':
            nonprofit_terms = ['nonprofit', 'fundraising', 'program', 'community', 'volunteer']
            if not any(term in resume_sections.get('experience', '').lower() for term in nonprofit_terms):
                feedback.append("Add program management experience, fundraising achievements, and community impact metrics.")
            else:
                feedback.append("Your nonprofit experience is clear. Highlight more program outcomes.")

        elif main_category == 'government':
            gov_terms = ['government', 'public', 'policy', 'administration', 'regulatory']
            if not any(term in resume_sections.get('experience', '').lower() for term in gov_terms):
                feedback.append("Include public service experience, policy implementation, and regulatory compliance.")
            else:
                feedback.append("Your government experience is evident. Add more policy impact metrics.")

        elif main_category == 'transportation':
            transport_terms = ['transportation', 'logistics', 'fleet', 'shipping', 'delivery']
            if not any(term in resume_sections.get('experience', '').lower() for term in transport_terms):
                feedback.append("Add logistics management, fleet operations, and delivery performance metrics.")
            else:
                feedback.append("Your transportation experience is clear. Consider highlighting efficiency metrics.")

        elif main_category == 'agriculture':
            ag_terms = ['agriculture', 'farming', 'crop', 'livestock', 'production']
            if not any(term in resume_sections.get('experience', '').lower() for term in ag_terms):
                feedback.append("Include agricultural production metrics, crop/livestock management, and yield improvements.")
            else:
                feedback.append("Your agricultural experience is evident. Add more production metrics.")

        elif main_category == 'fitness':
            fitness_terms = ['fitness', 'training', 'health', 'wellness', 'coaching']
            if not any(term in resume_sections.get('experience', '').lower() for term in fitness_terms):
                feedback.append("Add client training achievements, program development, and fitness assessment experience.")
            else:
                feedback.append("Your fitness experience is clear. Highlight more client success metrics.")

    # Add domain-specific feedback
    if job_analysis['domains']:
        # Add domain-specific recommendations based on identified domains
        for domain in job_analysis['domains']:
            if domain == 'software':
                feedback.append("Consider highlighting your software development methodologies and technical stack.")
            elif domain == 'data':
                feedback.append("Emphasize your data analysis tools and quantifiable project outcomes.")
            elif domain == 'cloud':
                feedback.append("Highlight your experience with specific cloud platforms and architectures.")

    # Add seniority-specific feedback
    if job_analysis['seniority'] == 'junior':
        feedback.append("Focus on your education, relevant coursework, and eagerness to learn.")
        
    elif job_analysis['seniority'] == 'senior':
        feedback.append("Emphasize your leadership experience and strategic contributions in previous roles.")
        
    elif job_analysis['seniority'] == 'manager':
        feedback.append("Highlight your experience managing teams, budgets, and complex projects.")
    
    # Add feedback based on evaluation metrics
    if evaluation_metrics['content_score'] < 3:
        feedback.append("Add more quantifiable achievements with specific metrics to strengthen your resume.")
        
    if evaluation_metrics['relevance_score'] < 3:
        # Get highly relevant skills for this job that might be missing
        missing_skills = []
        for skill in job_analysis['skills'][:3]:  # Get top 3 skills
            if skill not in resume_sections.get('skills', '').lower():
                missing_skills.append(skill)
                
        if missing_skills:
            feedback.append(f"Consider adding these relevant skills if you have experience with them: {', '.join(missing_skills)}.")
    
    # Add custom advice for specific job titles
    job_title_lower = job_title.lower()
    
    if 'developer' in job_title_lower or 'engineer' in job_title_lower:
        feedback.append("Showcase your problem-solving abilities with specific technical challenges you've overcome.")
        
    elif 'analyst' in job_title_lower:
        feedback.append("Highlight your analytical skills and experience working with data to drive decisions.")
        
    elif 'manager' in job_title_lower:
        feedback.append("Emphasize your leadership style and examples of successful team management.")
        
    elif 'designer' in job_title_lower:
        feedback.append("Demonstrate your design process and how your work has impacted user experience or business goals.")
    
    elif 'sales' in job_title_lower or 'business development' in job_title_lower:
        feedback.append("Quantify your sales achievements with specific revenue figures, growth percentages, and client acquisition metrics.")
    
    elif 'marketing' in job_title_lower:
        feedback.append("Showcase your campaign results with concrete metrics like ROI, engagement rates, and audience growth statistics.")
    
    elif 'data scientist' in job_title_lower:
        feedback.append("Detail your experience with specific machine learning models, data processing pipelines, and the business impact of your analyses.")
    
    elif 'product' in job_title_lower:
        feedback.append("Highlight your experience in product lifecycle management, user research, and cross-functional team collaboration.")
    
    elif 'doctor' in job_title_lower or 'physician' in job_title_lower:
        feedback.append("Emphasize your clinical expertise, patient care outcomes, and any specialized medical procedures or research.")
    
    elif 'nurse' in job_title_lower:
        feedback.append("Detail your experience with specific medical procedures, patient care metrics, and healthcare technology systems.")
    
    elif 'teacher' in job_title_lower or 'instructor' in job_title_lower:
        feedback.append("Highlight student achievement metrics, innovative teaching methods, and curriculum development experience.")
    
    elif 'lawyer' in job_title_lower or 'attorney' in job_title_lower:
        feedback.append("Showcase your case success rates, expertise in specific areas of law, and experience with notable legal proceedings.")
    
    elif 'accountant' in job_title_lower or 'finance' in job_title_lower:
        feedback.append("Emphasize your experience with specific financial software, regulatory compliance, and any cost-saving initiatives.")
    
    elif 'hr' in job_title_lower or 'human resources' in job_title_lower:
        feedback.append("Detail your experience with recruitment metrics, employee engagement initiatives, and HR policy development.")
    
    elif 'project manager' in job_title_lower:
        feedback.append("Quantify project outcomes, budget management success, and team size/scope of projects managed.")
    
    elif 'researcher' in job_title_lower or 'scientist' in job_title_lower:
        feedback.append("Highlight your publications, research methodologies, and the impact of your findings in your field.")
    
    elif 'content' in job_title_lower or 'writer' in job_title_lower:
        feedback.append("Showcase your content performance metrics, audience engagement, and experience with different content formats.")
    
    elif 'customer service' in job_title_lower or 'support' in job_title_lower:
        feedback.append("Emphasize your customer satisfaction scores, resolution rates, and experience with support tools/systems.")
    
    elif 'consultant' in job_title_lower:
        feedback.append("Detail specific client outcomes, ROI delivered, and expertise in particular industries or methodologies.")
    
    elif 'operations' in job_title_lower:
        feedback.append("Highlight process improvements, efficiency metrics, and experience with operations management tools.")
    
    elif 'chef' in job_title_lower or 'culinary' in job_title_lower:
        feedback.append("Showcase your menu development experience, kitchen management skills, and any signature dishes or specialties.")
    
    elif 'social media' in job_title_lower:
        feedback.append("Detail your experience with specific platforms, growth metrics, and successful social media campaigns.")
    
    elif 'security' in job_title_lower:
        feedback.append("Emphasize your experience with security protocols, incident response, and relevant certifications.")
    
    elif 'therapist' in job_title_lower or 'counselor' in job_title_lower:
        feedback.append("Highlight your counseling methodologies, types of cases handled, and any specialized therapeutic approaches.")
    
    # Combine feedback points into a cohesive paragraph
    combined_feedback = " ".join(feedback[:3])  # Limit to top 3 points for conciseness
    
    # Ensure proper formatting and sentence structure
    if not combined_feedback.endswith('.'):
        combined_feedback += '.'
        
    # Add a generic closing recommendation if feedback is too short
    if len(combined_feedback.split()) < 20:
        combined_feedback += f" To be more competitive for {job_title} positions, tailor your resume to highlight relevant experience and achievements that demonstrate your capability in this role."
    
    return combined_feedback

def analyze_ats_compatibility(text, job_title):
    """
    Analyze the resume for ATS (Applicant Tracking System) compatibility
    and provide recommendations for improvement.
    """
    ats_issues = []
    ats_score = 10  # Start with perfect score and deduct for issues
    
    # Check for common ATS issues
    
    # 1. Check for complex formatting (tables, columns, headers/footers) - indirect indicators
    unusual_whitespace = len(re.findall(r'\s{4,}', text))
    if unusual_whitespace > 10:
        ats_issues.append("Potential complex formatting detected. Ensure your resume uses a simple format without tables or multiple columns.")
        ats_score -= 2
    
    # 2. Check for uncommon section headings
    common_headings = ['experience', 'education', 'skills', 'summary', 'objective', 'projects']
    found_headings = []
    
    for line in text.split('\n'):
        line = line.strip().lower()
        if line and len(line) < 30 and line.endswith(':'):
            found_headings.append(line.rstrip(':'))
    
    uncommon_headings = [h for h in found_headings if not any(common in h for common in common_headings)]
    if uncommon_headings:
        ats_issues.append(f"Uncommon section headings detected: {', '.join(uncommon_headings)}. Consider using standard headings that ATS systems recognize.")
        ats_score -= len(uncommon_headings) * 0.5
    
    # 3. Check for job title keyword presence
    job_keywords = job_title.lower().split()
    matched_keywords = [kw for kw in job_keywords if kw in text.lower() and len(kw) > 3]
    
    if len(matched_keywords) < len(job_keywords) / 2:
        ats_issues.append(f"Your resume may not contain enough keywords related to '{job_title}'. Consider adding more relevant terms.")
        ats_score -= 2
    
    # 4. Check for contact information
    contact_patterns = {
        'email': r'[\w\.-]+@[\w\.-]+\.\w+',
        'phone': r'(\+\d{1,3}[-\.\s]??)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}'
    }
    
    missing_contact = []
    for contact_type, pattern in contact_patterns.items():
        if not re.search(pattern, text):
            missing_contact.append(contact_type)
    
    if missing_contact:
        ats_issues.append(f"Missing or non-standard {'/'.join(missing_contact)} format. Ensure your contact information is clearly visible.")
        ats_score -= len(missing_contact)
    
    # 5. Check for file type indicators
    file_type_issues = []
    if ".pdf" in text or ".docx" in text or ".doc" in text:
        file_type_issues.append("Document may contain visible file extensions, which could confuse ATS systems.")
        ats_score -= 1
    
    if "Page" in text and re.search(r'Page\s+\d+\s+of\s+\d+', text):
        file_type_issues.append("Document appears to contain page numbers, which may interfere with ATS parsing.")
        ats_score -= 1
    
    if file_type_issues:
        ats_issues.extend(file_type_issues)
    
    # Normalize ATS score
    ats_score = max(1, min(10, ats_score))
    
    return {
        "ats_score": ats_score,
        "ats_issues": ats_issues[:5],  # Limit to top 5 issues
        "ats_compatible": ats_score >= 7
    }

def extract_skills_from_text(skills_text):
    """
    Extract actual skills from skills section text, filtering out
    non-skill items like location names, common words, etc.
    """
    if not skills_text:
        return []
        
    # Common technical skills - we'll use this to validate extracted skills
    common_tech_skills = [
        # Programming Languages
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'php', 'swift', 'kotlin', 'rust',
        'scala', 'r', 'matlab', 'perl', 'haskell', 'lua', 'dart', 'groovy', 'objective-c', 'assembly', 'cobol',
        'fortran', 'vba', 'shell scripting', 'powershell', 'bash', 'tcl', 'erlang', 'clojure', 'f#', 'julia',
        
        # Web Development - Frontend
        'html', 'css', 'sass', 'less', 'javascript', 'typescript', 'react', 'angular', 'vue.js', 'svelte',
        'jquery', 'bootstrap', 'material-ui', 'tailwind css', 'webpack', 'babel', 'redux', 'vuex', 'next.js',
        'nuxt.js', 'gatsby', 'ember.js', 'backbone.js', 'three.js', 'd3.js', 'webgl', 'web components',
        'progressive web apps', 'service workers', 'web sockets', 'web assembly', 'web workers',
        'responsive design', 'cross-browser compatibility', 'web accessibility', 'web performance optimization',
        
        # Web Development - Backend
        'node.js', 'express.js', 'django', 'flask', 'spring', 'spring boot', 'ruby on rails', 'laravel',
        'asp.net core', 'fastapi', 'nest.js', 'graphql', 'rest apis', 'soap', 'grpc', 'websockets',
        'microservices', 'oauth', 'jwt', 'web security', 'cors', 'rate limiting', 'caching strategies',
        
        # Databases & Data Storage
        'sql', 'mysql', 'postgresql', 'oracle', 'sql server', 'sqlite', 'mongodb', 'cassandra', 'redis',
        'elasticsearch', 'dynamodb', 'couchbase', 'neo4j', 'influxdb', 'mariadb', 'firebase', 'supabase',
        'cockroachdb', 'timescaledb', 'clickhouse', 'snowflake', 'big query', 'redshift', 'data warehousing',
        'etl', 'data modeling', 'database optimization', 'database administration', 'data migration',
        
        # Cloud Computing & DevOps
        'aws', 'azure', 'google cloud platform', 'alibaba cloud', 'ibm cloud', 'oracle cloud', 'digitalocean',
        'docker', 'kubernetes', 'terraform', 'ansible', 'chef', 'puppet', 'jenkins', 'gitlab ci', 'github actions',
        'circleci', 'travis ci', 'prometheus', 'grafana', 'elk stack', 'nginx', 'apache', 'load balancing',
        'auto scaling', 'serverless', 'lambda functions', 'cloud formation', 'azure resource manager',
        'infrastructure as code', 'configuration management', 'service mesh', 'istio', 'envoy',
        
        # Cloud Services & Tools
        'ec2', 's3', 'rds', 'dynamodb', 'sqs', 'sns', 'cloudfront', 'route 53', 'vpc', 'iam',
        'azure vm', 'azure storage', 'azure functions', 'cosmos db', 'azure devops', 'app service',
        'google compute engine', 'google kubernetes engine', 'cloud storage', 'cloud functions',
        'cloud pub/sub', 'big query', 'cloud spanner', 'firebase', 'heroku', 'netlify', 'vercel',
        
        # Data Science & Machine Learning
        'machine learning', 'deep learning', 'artificial intelligence', 'neural networks', 'computer vision',
        'natural language processing', 'reinforcement learning', 'data mining', 'statistical analysis',
        'predictive modeling', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
        'scipy', 'matplotlib', 'seaborn', 'plotly', 'jupyter', 'opencv', 'nltk', 'spacy', 'gensim',
        'xgboost', 'lightgbm', 'catboost', 'h2o', 'mlflow', 'kubeflow', 'data version control',
        
        # Big Data Technologies
        'hadoop', 'spark', 'hive', 'pig', 'kafka', 'storm', 'flink', 'cassandra', 'hbase',
        'data lakes', 'data warehousing', 'etl pipelines', 'data streaming', 'real-time analytics',
        'batch processing', 'distributed computing', 'map reduce', 'yarn', 'zookeeper', 'airflow',
        'nifi', 'talend', 'informatica', 'databricks', 'snowflake', 'redshift', 'big query',
        
        # Cybersecurity
        'network security', 'application security', 'cloud security', 'security architecture',
        'penetration testing', 'vulnerability assessment', 'ethical hacking', 'cryptography',
        'security protocols', 'firewall configuration', 'intrusion detection', 'incident response',
        'security information and event management (siem)', 'identity and access management',
        'security automation', 'threat intelligence', 'malware analysis', 'forensics',
        'devsecops', 'zero trust architecture', 'compliance frameworks', 'security auditing',
        
        # Mobile Development
        'ios development', 'android development', 'react native', 'flutter', 'xamarin',
        'swift', 'objective-c', 'kotlin', 'java android', 'mobile ui/ux', 'responsive design',
        'progressive web apps', 'ionic', 'cordova', 'mobile security', 'app store optimization',
        'mobile analytics', 'push notifications', 'mobile testing', 'mobile performance optimization',
        
        # Testing & Quality Assurance
        'unit testing', 'integration testing', 'functional testing', 'automated testing',
        'performance testing', 'load testing', 'stress testing', 'security testing',
        'test automation', 'selenium', 'appium', 'junit', 'pytest', 'jest', 'mocha',
        'cypress', 'postman', 'swagger', 'test case design', 'bug tracking', 'quality metrics',
        'continuous testing', 'test-driven development', 'behavior-driven development',
        
        # Development Tools & Practices
        'git', 'svn', 'mercurial', 'jira', 'confluence', 'trello', 'agile', 'scrum',
        'kanban', 'waterfall', 'code review', 'pair programming', 'continuous integration',
        'continuous deployment', 'version control', 'documentation', 'api design',
        'software architecture', 'design patterns', 'clean code', 'refactoring',
        'performance optimization', 'debugging', 'monitoring', 'logging',
        
        # Emerging Technologies
        'blockchain', 'smart contracts', 'ethereum', 'solidity', 'web3', 'nft',
        'augmented reality', 'virtual reality', 'mixed reality', 'internet of things',
        'edge computing', 'quantum computing', 'robotics', '5g', 'computer vision',
        'speech recognition', 'natural language generation', 'autonomous systems',
        'digital twins', 'metaverse', 'cryptocurrency', 'distributed ledger',
        
        # Business Intelligence & Analytics
        'tableau', 'power bi', 'qlik', 'looker', 'sisense', 'data visualization',
        'business analytics', 'data analysis', 'statistical analysis', 'predictive analytics',
        'prescriptive analytics', 'data modeling', 'data warehousing', 'olap',
        'reporting tools', 'dashboard design', 'kpi monitoring', 'data governance',
        'data quality management', 'master data management', 'business intelligence',
        
        # Project Management & Collaboration
        'project management', 'agile methodologies', 'scrum master', 'product owner',
        'sprint planning', 'backlog management', 'risk management', 'stakeholder management',
        'resource allocation', 'budgeting', 'time tracking', 'project scheduling',
        'team collaboration', 'remote work tools', 'virtual team management',
        'project documentation', 'change management', 'release management',
        
        # System Administration
        'linux administration', 'windows server', 'active directory', 'dns', 'dhcp',
        'network administration', 'system monitoring', 'backup and recovery',
        'disaster recovery', 'high availability', 'load balancing', 'virtualization',
        'vmware', 'hyper-v', 'server maintenance', 'patch management',
        'system security', 'performance tuning', 'troubleshooting',
        
        # UI/UX Design
        'user interface design', 'user experience design', 'wireframing', 'prototyping',
        'figma', 'sketch', 'adobe xd', 'invision', 'zeplin', 'principle',
        'user research', 'usability testing', 'interaction design', 'visual design',
        'information architecture', 'accessibility design', 'responsive design',
        'mobile-first design', 'design systems', 'design thinking',

        # Healthcare & Medical
        'patient care', 'vital signs monitoring', 'medical terminology', 'electronic health records (ehr)',
        'hipaa compliance', 'medical coding', 'clinical documentation', 'infection control',
        'patient assessment', 'medication administration', 'wound care', 'phlebotomy',
        'diagnostic procedures', 'medical imaging', 'laboratory testing', 'emergency medicine',
        'telemedicine', 'patient education', 'medical device operation', 'healthcare regulations',
        'medical billing', 'anatomy knowledge', 'physiology', 'pharmacology', 'nursing care plans',
        'surgical procedures', 'mental health assessment', 'rehabilitation therapy', 'immunizations',
        'medical research', 'clinical trials', 'disease management', 'preventive care',

        # Business & Finance
        'financial analysis', 'market research', 'strategic planning', 'business development',
        'account management', 'sales strategies', 'customer relationship management (crm)',
        'financial modeling', 'business intelligence', 'profit & loss management',
        'business process improvement', 'contract negotiation', 'revenue forecasting',
        'cost analysis', 'budget management', 'investment analysis', 'risk assessment',
        'mergers & acquisitions', 'business valuation', 'financial reporting',
        'tax planning', 'audit procedures', 'regulatory compliance', 'business law',
        'supply chain management', 'inventory management', 'vendor relations',
        'business strategy', 'market analysis', 'competitive analysis', 'pricing strategies',

        # Education & Teaching
        'curriculum development', 'lesson planning', 'student assessment', 'classroom management',
        'educational technology', 'differentiated instruction', 'special education',
        'student engagement', 'behavior management', 'educational psychology',
        'learning theories', 'instructional design', 'distance learning', 'e-learning',
        'student counseling', 'academic advising', 'educational leadership',
        'standardized testing', 'individualized education plans (iep)', 'gifted education',
        'early childhood education', 'adult education', 'stem education', 'literacy instruction',
        'educational assessment', 'student motivation', 'parent communication', 'teaching methods',

        # Legal & Compliance
        'legal research', 'contract law', 'litigation', 'regulatory compliance',
        'legal documentation', 'case management', 'intellectual property law',
        'corporate law', 'employment law', 'legal writing', 'legal analysis',
        'due diligence', 'risk management', 'negotiations', 'mediation',
        'legal ethics', 'court procedures', 'legal consultation', 'policy development',
        'compliance training', 'regulatory reporting', 'legal technology',
        'document review', 'legal research tools', 'legal project management',
        'administrative law', 'civil litigation', 'criminal law', 'family law',

        # Marketing & Communications
        'marketing strategy', 'brand management', 'social media marketing',
        'content marketing', 'digital marketing', 'email marketing', 'seo/sem',
        'public relations', 'advertising', 'market research', 'campaign management',
        'copywriting', 'brand development', 'marketing analytics', 'media planning',
        'event planning', 'crisis communication', 'community management',
        'influencer marketing', 'marketing automation', 'customer segmentation',
        'competitive analysis', 'marketing roi', 'brand storytelling',
        'marketing communications', 'presentation skills', 'public speaking',

        # Human Resources
        'recruitment', 'talent acquisition', 'employee relations', 'performance management',
        'compensation & benefits', 'hr policies', 'workforce planning', 'training & development',
        'succession planning', 'employee engagement', 'labor relations', 'hr analytics',
        'organizational development', 'diversity & inclusion', 'employee onboarding',
        'hr compliance', 'benefits administration', 'employee wellness', 'conflict resolution',
        'hr information systems', 'payroll management', 'workplace safety',
        'employee retention', 'hr strategy', 'talent management', 'change management',

        # Customer Service & Support
        'customer support', 'conflict resolution', 'problem solving', 'service delivery',
        'customer satisfaction', 'quality assurance', 'complaint resolution',
        'call center operations', 'customer feedback', 'service level agreements',
        'technical support', 'help desk management', 'customer experience',
        'service optimization', 'customer retention', 'account management',
        'client relations', 'service metrics', 'customer service tools',
        'escalation management', 'customer analytics', 'service standards',

        # Research & Analysis
        'research methodology', 'data collection', 'quantitative analysis',
        'qualitative analysis', 'research design', 'statistical analysis',
        'survey design', 'focus groups', 'research ethics', 'literature review',
        'experimental design', 'research validation', 'data interpretation',
        'research documentation', 'hypothesis testing', 'research presentation',
        'academic writing', 'grant writing', 'peer review', 'publication',
        'research compliance', 'research coordination', 'clinical research',

        # Hospitality & Tourism
        'hotel management', 'restaurant operations', 'event planning',
        'hospitality marketing', 'guest services', 'food & beverage management',
        'tourism development', 'reservation systems', 'revenue management',
        'hospitality law', 'customer service', 'facility management',
        'menu planning', 'inventory control', 'housekeeping operations',
        'front desk operations', 'concierge services', 'travel planning',
        'tourism marketing', 'hospitality technology', 'guest relations',

        # Social Services
        'case management', 'counseling', 'social work', 'community outreach',
        'crisis intervention', 'program development', 'advocacy', 'mental health',
        'family services', 'substance abuse', 'behavioral health', 'group facilitation',
        'needs assessment', 'resource coordination', 'intervention planning',
        'social service programs', 'community resources', 'support services',
        'rehabilitation services', 'youth services', 'elder care', 'disability services',

        # Environmental & Sustainability
        'environmental assessment', 'sustainability planning', 'environmental compliance',
        'waste management', 'renewable energy', 'environmental impact analysis',
        'conservation', 'environmental monitoring', 'green building', 'carbon footprint',
        'environmental regulations', 'sustainable development', 'energy efficiency',
        'environmental protection', 'natural resource management', 'pollution control',
        'environmental education', 'climate change', 'biodiversity', 'eco-friendly practices',

        # Retail & Sales
        'retail management', 'sales techniques', 'merchandising', 'inventory management',
        'point of sale systems', 'retail operations', 'sales forecasting',
        'customer service', 'visual merchandising', 'retail marketing',
        'store operations', 'loss prevention', 'category management',
        'pricing strategies', 'sales analytics', 'retail technology',
        'customer experience', 'brand representation', 'sales training',
        'retail planning', 'market analysis', 'product knowledge'
    ]
    
    # Try to extract skills using a pattern
    skills_pattern = r'(?:^|,|\n)\s*([A-Za-z0-9#\+\-\s\/\.]{2,30}?)(?:$|,|\n)'
    skills_matches = re.findall(skills_pattern, skills_text)
    
    # Process and filter the raw skill matches
    valid_skills = []
    for skill in skills_matches:
        # Clean up the skill text
        skill = skill.strip().lower()
        
        # Skip if too short or empty
        if len(skill) < 2:
            continue
            
        # Skip if it's likely not a skill (e.g., location names, common words)
        if (
            # Check if it's a common skill or contains a common skill term
            any(tech.lower() in skill for tech in common_tech_skills) or
            
            # Check for well-known technical abbreviations (3+ uppercase chars)
            re.search(r'\b[A-Z]{3,}\b', skill) or
            
            # Contains technical terms or numbers+letters (likely technical)
            re.search(r'\b(?:framework|language|platform|tool|software|hardware|system)\b', skill) or
            re.search(r'\b[A-Za-z]+[0-9]+[A-Za-z]*\b', skill)
        ):
            valid_skills.append(skill)
    
    # If we couldn't find many valid skills, try a different approach
    if len(valid_skills) < 3:
        # Look for nouns and noun phrases in the skills section
        doc = nlp(skills_text[:1000])  # Process first 1000 chars for efficiency
        noun_chunks = [chunk.text.lower().strip() for chunk in doc.noun_chunks if 2 < len(chunk.text) < 30]
        valid_skills.extend(noun_chunks)
        
        # Also include any technical terms and abbreviations
        tech_terms = re.findall(r'\b[A-Z][A-Za-z0-9\+\#]+\b', skills_text)
        valid_skills.extend([term.lower() for term in tech_terms])
    
    # Remove duplicates and sort
    valid_skills = list(set(valid_skills))
    
    return valid_skills

def extract_positions(experience_text):
    """
    Extract job positions/titles from experience text using regex patterns.
    Returns a list of identified positions.
    """
    if not experience_text:
        return []
    position_pattern = r'(?:^|\n)([A-Z][a-zA-Z\s]+)(?=\s*\b(?:at|,|\-|\(|from|to)\b)'
    positions = re.findall(position_pattern, experience_text)
    return [pos.strip() for pos in positions if len(pos.strip()) > 3]

def extract_degrees(education_text):
    """
    Extract education degrees from education text using regex patterns.
    Returns a list of identified degrees.
    """
    if not education_text:
        return []
    degree_pattern = r'\b(?:Bachelor|Master|MBA|PhD|BS|BA|MS|MA|Doctor|Associate)\b(?:[^\n.]*)'
    degrees = re.findall(degree_pattern, education_text, re.IGNORECASE)
    return [deg.strip() for deg in degrees if len(deg.strip()) > 1]
    
# Update analyze_resume to incorporate job description matching and achievement examples
def analyze_resume(file, job_title, industry=None, custom_keywords=None, job_description=None):
    if not file:
        return "⚠️ Please upload a resume file."
    
    text, error = extract_text(file)
    if not text:
        return error
    
    # Extract resume sections for detailed analysis
    resume_sections = extract_resume_sections(text)
    
    # Perform comprehensive evaluation
    evaluation = comprehensive_resume_evaluation(text, job_title, resume_sections)
    
    # Add ATS analysis
    ats_analysis = analyze_ats_compatibility(text, job_title)
    
    # Get job-specific recommendations
    job_recommendations = get_advanced_job_recommendations(job_title, industry)
    
    # Generate role-specific feedback using the language model
    role_feedback = generate_role_specific_feedback(job_title, resume_sections, evaluation)
    
    # Analyze match against job description if provided
    job_match = None
    if job_description and len(job_description) > 100:
        job_match = analyze_resume_job_match(text, job_description, job_title)
    
    # Get job title analysis for dynamic evaluations
    job_analysis = analyze_job_title(job_title)
    
    # Detect if this is a student resume
    is_student = False
    if 'education' in resume_sections:
        is_student = re.search(r'(current|enrolled|attending|studying|expected grad|to graduate|in progress)', 
                             resume_sections['education'], re.IGNORECASE) is not None
    
    # Format the results in a detailed, structured report
    result = f"""### Resume Analysis Report for {job_title} Position

**Overall Rating:** {evaluation['overall_score']:.1f}/10
**ATS Compatibility Score:** {ats_analysis['ats_score']:.1f}/10 {' ✅' if ats_analysis['ats_compatible'] else ' ⚠️'}
"""

    # Add job description match score if available
    if job_match and job_match["has_job_description"]:
        result += f"**Job Match Score:** {job_match['match_score']}% match with provided job description\n"

    result += """
#### Category Scores:
- **Format & Structure:** {0:.1f}/5
- **Content Quality:** {1:.1f}/5
- **Job Relevance:** {2:.1f}/5
- **Completeness:** {3:.1f}/5
- **Clarity & Impact:** {4:.1f}/5

#### Key Strengths:
""".format(
        evaluation['format_score'],
        evaluation['content_score'],
        evaluation['relevance_score'],
        evaluation['completeness_score'],
        evaluation['clarity_score']
    )
    
    for strength in evaluation['strengths']:
        result += f"- {strength}\n"
    
    result += "\n#### Areas for Improvement:\n"
    for improvement in evaluation['improvements']:
        result += f"- {improvement}\n"
    
    # Add job description match analysis if available
    if job_match and job_match["has_job_description"]:
        result += "\n#### Job Description Match Analysis:\n"
        result += f"- Your resume matches {job_match['match_score']}% of keywords found in the job description.\n"
        
        if job_match["matching_keywords"]:
            result += "- **Matching Keywords:** " + ", ".join(job_match["matching_keywords"]) + "\n"
        
        if job_match["missing_keywords"]:
            result += "- **Missing Keywords:** " + ", ".join(job_match["missing_keywords"]) + "\n"
            result += "- Consider adding these missing keywords to improve your match rate.\n"
    
    # Add role-specific feedback
    result += f"""
#### Relevance to {job_title} Position:
The resume shows {'strong' if evaluation['relevance_score'] > 3 else 'moderate' if evaluation['relevance_score'] > 1 else 'limited'} alignment with the target role.

#### Role-Specific Feedback:
{role_feedback}
"""
    
    # Add achievement examples if available
    if 'achievements' in evaluation and evaluation['achievements']:
        result += "\n#### Achievement Examples Detected:\n"
        for i, achievement in enumerate(evaluation['achievements']):
            result += f"- {achievement}\n"
        
        if len(evaluation['achievements']) < 3:
            result += f"\n**Tip:** Try to include more measurable achievements. For {job_title} roles, quantify your impact whenever possible.\n"
    
    # Add ATS compatibility information
    result += "\n#### ATS Compatibility:\n"
    if ats_analysis['ats_issues']:
        for issue in ats_analysis['ats_issues']:
            result += f"- {issue}\n"
    else:
        result += "- Your resume appears to be ATS-friendly. Continue to ensure standard formatting and clear section headings.\n"
    
    # Add section-specific feedback if sections exist
    if 'experience' in resume_sections:
        result += "\n#### Experience Analysis:\n"
        positions = extract_positions(resume_sections['experience'])
        if positions:
            result += f"- Positions identified: {', '.join(positions[:3])}" + (f" and {len(positions)-3} more" if len(positions) > 3 else "") + "\n"
        else:
            result += "- Consider expanding your work experience with more specific achievements and responsibilities.\n"
    
    if 'education' in resume_sections:
        result += "\n#### Education Analysis:\n"
        degrees = extract_degrees(resume_sections['education'])
        if degrees:
            result += f"- Education credentials identified: {', '.join(degrees[:2])}" + (f" and more" if len(degrees) > 2 else "") + "\n"
        else:
            result += "- Consider formatting your education section to clearly highlight degrees and institutions.\n"
        
    if 'skills' in resume_sections:
        result += "\n#### Skills Analysis:\n"
        skill_length = len(resume_sections['skills'].split())
        if skill_length > 50:
            result += "- Comprehensive skills section with good detail.\n"
        else:
            result += "- Consider expanding your skills section with more specific technical and soft skills.\n"
    
    # Check for missing essential sections
    essential_sections = {'summary', 'experience', 'education', 'skills'}
    missing_sections = essential_sections - set(resume_sections.keys())
    if missing_sections:
        result += "\n## Missing Sections\n"
        for section in missing_sections:
            result += f"⚠ Consider adding {section} section\n"
    
    word_count = len(text.split())
    result += f"- Resume length: {word_count} words ({word_count/250:.1f} pages equivalent).\n"
    
    return result

# Update the process_resume function to handle the new parameters
def process_resume(file, job_title, industry=None, custom_keywords=None, job_description=None):
    return analyze_resume(file, job_title, industry, custom_keywords, job_description)

# Gradio UI with improved modern design
with gr.Blocks(title="AI Resume Evaluator Pro", theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate")) as ui:
    # Header with logo-like design
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Image(value="sm_logo.png", show_label=False, container=False, height=80, width=80)
        with gr.Column(scale=5):
            gr.HTML("""
                <div style="text-align: center;">
                    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700; background: linear-gradient(90deg, #4338ca, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Santek AI Resume Evaluator Pro</h1>
                    <p style="margin-top: 0; font-size: 1.1rem; color: #6b7280;">Upload your resume and get AI-powered feedback tailored to your target role</p>
                </div>
            """)

    gr.HTML("""<div style="height: 1px; background: linear-gradient(90deg, rgba(255,255,255,0), rgba(120,120,120,0.2), rgba(255,255,255,0)); margin: 15px 0;"></div>""")
    
    # Main content area with improved layout
    with gr.Row():
        # Left side: Inputs with improved styling
        with gr.Column(scale=3):
            with gr.Group(elem_classes="input-container"):
                gr.HTML("""<h3 style="margin-top: 0; font-size: 1.3rem; color: #4338ca;">📄 Upload & Configure</h3>""")
                
                file_input = gr.File(
                    label="Upload Resume",
                    file_types=[".pdf", ".docx"],
                    elem_id="file_upload"
                )
                
                gr.HTML("""<div style="font-size: 0.85rem; color: #6b7280; margin-bottom: 15px;">
                    Supported formats: PDF, DOCX. Files are processed securely and not stored.
                </div>""")
                
                job_title = gr.Textbox(
                    label="Target Job Title",
                    placeholder="e.g., Data Scientist, Marketing Manager, Cloud Engineer",
                    info="Be specific for better results"
                )
                
                with gr.Accordion("✨ Advanced Options", open=False):
                    industry_selector = gr.Dropdown(
                        choices=["Technology", "Healthcare", "Finance", "Education", "Marketing", 
                                "Legal", "Engineering", "Sales", "Design", "Management", "Other"],
                        label="Industry",
                        value="Technology",
                        info="Select the industry for more targeted analysis"
                    )
                    
                    custom_keywords = gr.Textbox(
                        label="Custom Keywords",
                        placeholder="Enter comma-separated keywords specific to your target job",
                        info="Keywords that are particularly important for the role"
                    )
                    
                    job_description = gr.Textbox(
                        label="Job Description",
                        placeholder="Paste the full job description here for better matching analysis",
                        lines=6,
                        info="Adding a job description significantly improves the analysis accuracy"
                    )
            
            # Submit button with improved styling
            submit_btn = gr.Button(
                "🔍 Analyze My Resume", 
                variant="primary",
                size="lg"
            )
            
            # Help panel with improved visual design
            with gr.Accordion("📊 How It Works", open=False):
                gr.HTML("""
                <div style="padding: 10px 0;">
                    <div style="display: flex; margin-bottom: 15px;">
                        <div style="min-width: 40px; font-size: 25px; color: #4338ca; text-align: center;">1</div>
                        <div>
                            <h4 style="margin: 0; color: #1e3a8a;">Upload & Configure</h4>
                            <p style="margin: 5px 0 0 0; color: #6b7280;">Upload your resume and enter your target job title. For best results, paste the job description too.</p>
                        </div>
                    </div>
                    
                    <div style="display: flex; margin-bottom: 15px;">
                        <div style="min-width: 40px; font-size: 25px; color: #4338ca; text-align: center;">2</div>
                        <div>
                            <h4 style="margin: 0; color: #1e3a8a;">AI Analysis</h4>
                            <p style="margin: 5px 0 0 0; color: #6b7280;">Our AI analyzes your resume for format, content, relevance, and ATS compatibility.</p>
                        </div>
                    </div>
                    
                    <div style="display: flex;">
                        <div style="min-width: 40px; font-size: 25px; color: #4338ca; text-align: center;">3</div>
                        <div>
                            <h4 style="margin: 0; color: #1e3a8a;">Actionable Feedback</h4>
                            <p style="margin: 5px 0 0 0; color: #6b7280;">Get detailed scores, strengths, improvements, and role-specific recommendations.</p>
                        </div>
                    </div>
                </div>
                """)
        
        # Right side: Output with improved styling
        with gr.Column(scale=4):
            with gr.Group(elem_classes="output-container"):
                gr.HTML("""<h3 style="margin-top: 0; font-size: 1.3rem; color: #4338ca;">🔬 Analysis Results</h3>""")
                output = gr.Markdown(label="Analysis Results", elem_id="result_panel")
                gr.HTML("""
                <div id="waiting_message" style="text-align: center; color: #6b7280; padding: 50px 20px; display: none;">
                    <div style="margin-bottom: 20px;">
                        <img src="https://img.icons8.com/color/96/000000/search-in-browser.png" width="60" height="60">
                    </div>
                    <div id="analysis_status" style="font-size: 1.2rem; margin-bottom: 10px;">
                        Waiting for resume...
                    </div>
                    <div style="font-size: 0.9rem;">
                        Upload your resume and click "Analyze My Resume" to get started
                    </div>
                </div>
                
                <script>
                    // Show waiting message when output is empty
                    document.addEventListener('DOMContentLoaded', function() {
                        const resultPanel = document.getElementById('result_panel');
                        const waitingMessage = document.getElementById('waiting_message');
                        
                        // Initial check
                        if (!resultPanel.textContent.trim()) {
                            waitingMessage.style.display = 'block';
                        }
                        
                        // Set up observer to monitor changes
                        const observer = new MutationObserver(function(mutations) {
                            if (resultPanel.textContent.trim()) {
                                waitingMessage.style.display = 'none';
                            } else {
                                waitingMessage.style.display = 'block';
                            }
                        });
                        
                        observer.observe(resultPanel, { 
                            childList: true,
                            characterData: true,
                            subtree: true 
                        });
                        
                        // Update status when file is uploaded
                        const fileUpload = document.getElementById('file_upload');
                        if (fileUpload) {
                            fileUpload.addEventListener('change', function() {
                                const status = document.getElementById('analysis_status');
                                if (status) status.textContent = 'Resume ready! Click "Analyze My Resume"';
                            });
                        }
                    });
                </script>
                """)
    
    # Tips section with visually appealing design
    with gr.Accordion("💡 Resume Tips & Best Practices", open=False):
        gr.HTML("""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 10px;">
            <div style="background: linear-gradient(to bottom right, #4338ca, #6366f1); border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: white;">🎯 Tailoring Tips</h4>
                <ul style="margin: 0; padding-left: 20px; color: white;">
                    <li>Customize your resume for each specific job application</li>
                    <li>Mirror language from the job description</li>
                    <li>Prioritize relevant experience and skills</li>
                    <li>Include industry-specific keywords</li>
                </ul>
            </div>
            
            <div style="background: linear-gradient(to bottom right, #1e40af, #3b82f6); border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: white;">📊 Content Tips</h4>
                <ul style="margin: 0; padding-left: 20px; color: white;">
                    <li>Quantify achievements with numbers, percentages, and metrics</li>
                    <li>Use strong action verbs to begin bullet points</li>
                    <li>Focus on accomplishments rather than just responsibilities</li>
                    <li>Include relevant projects and certifications</li>
                </ul>
            </div>
            
            <div style="background: linear-gradient(to bottom right, #0369a1, #0ea5e9); border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: white;">🤖 ATS Tips</h4>
                <ul style="margin: 0; padding-left: 20px; color: white;">
                    <li>Use a simple, clean format with standard section headings</li>
                    <li>Avoid tables, graphics, headers/footers and complex formatting</li>
                    <li>Include exact keywords from the job description</li>
                    <li>Use standard file formats like .docx or PDF</li>
                </ul>
            </div>
        </div>
        """)
    
    # Footer with disclaimer and branding
    gr.HTML("""
    <div style="margin-top: 30px; text-align: center; border-top: 1px solid #e5e7eb; padding-top: 15px;">
        <div style="font-size: 0.85rem; color: #6b7280;">
            AI Resume Evaluator Pro | Your resume data is processed locally and not stored
        </div>
        <div style="font-size: 0.85rem; color: #6b7280; margin-top: 5px;">
            Owned by <a href="https://www.santekmicrosolutions.com/" target="_blank" style="color: #4338ca; text-decoration: none;">Santek Micro Solutions</a> | IT Services and IT Consulting | Phone: <a href="tel:4169518616" style="color: #4338ca; text-decoration: none;">416-951-8616</a>
        </div>
    </div>
    """)
    
    # Add custom CSS for additional styling
    gr.HTML("""
    <style>
        /* General improvements */
        .gradio-container {
            max-width: 1200px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
        
        /* Improved button hover effect */
        button.primary {
            transition: all 0.2s ease !important;
        }
        button.primary:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(67, 56, 202, 0.15) !important;
        }
        
        /* Styling for the containers that were using Box */
        .input-container, .output-container {
            border: 1px solid #2d3748;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 15px;
            background-color: #1e293b;
            color: #f8fafc;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2), 0 1px 2px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
        }
        .input-container:hover, .output-container:hover {
            box-shadow: 0 4px 10px rgba(0,0,0,0.1), 0 4px 5px rgba(0,0,0,0.1);
        }
        
        /* Improve text color for better contrast against dark background */
        .input-container h3, .output-container h3 {
            color: #f0f9ff !important;
        }
        
        .input-container label, .output-container label,
        .input-container p, .output-container p {
            color: #e2e8f0 !important;
        }
        
        /* Style input elements to integrate better with container */
        .input-container input, 
        .input-container textarea, 
        .input-container select {
            background-color: #f0f4f8 !important;
            border: 1px solid #475569 !important;
            border-radius: 6px !important;
            margin: 2px 0 !important;
            color: #1e293b !important; /* Dark text for light backgrounds */
        }
        
        /* Fix dropdown styling */
        select.gr-box, div[class*="dropdown"] button, 
        div[class*="dropdown"] ul, div[class*="dropdown"] li {
            color: #1e293b !important; /* Dark text */
            background-color: #f0f4f8 !important; /* Light background */
        }
        
        /* Ensure dropdown options are visible */
        div[class*="dropdown"] ul {
            background-color: white !important;
            color: #1e293b !important;
            border: 1px solid #475569 !important;
        }
        
        div[class*="dropdown"] li:hover {
            background-color: #e2e8f0 !important;
        }
        
        /* Add subtle inner shadow for depth */
        .input-container input:focus, 
        .input-container textarea:focus, 
        .input-container select:focus {
            background-color: #f8fafc !important;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05), 0 0 0 1px rgba(67, 56, 202, 0.2) !important;
        }
        
        /* Remove excess padding from Gradio elements */
        .gradio-container .prose {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }
        
        .gradio-container .form {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding: 0 !important;
        }
        
        /* Accordion animations */
        .gr-accordion {
            transition: all 0.3s ease !important;
        }
        
        /* Input focus effects */
        input:focus, textarea:focus, select:focus {
            border-color: #4338ca !important;
            box-shadow: 0 0 0 1px rgba(67, 56, 202, 0.2) !important;
        }
        
        /* Fix dropdown text colors */
        .input-container select {
            color: #0f172a !important; 
            background-color: #f0f4f8 !important;
        }
        
        /* Style dropdown options */
        select option {
            color: #0f172a !important;
            background-color: #f8fafc !important;
        }
        
        /* Fix text inside other form elements */
        .input-container .gr-dropdown {
            color: #0f172a !important;
            background-color: #f0f4f8 !important;
        }
        
        /* Make Gradio dropdown trigger visible */
        .gr-dropdown > button, 
        .gr-dropdown > div button {
            color: #0f172a !important;
            background-color: #f0f4f8 !important;
        }
        
        /* Style dropdown menu items for Gradio components */
        .gr-dropdown > div,
        .gr-dropdown ul li button {
            color: #0f172a !important;
            background-color: #f8fafc !important;
        }
    </style>
    """)

    # Connect the submit button to the process_resume function
    submit_btn.click(
        fn=process_resume, 
        inputs=[file_input, job_title, industry_selector, custom_keywords, job_description], 
        outputs=output
    )

# Launch the app with public sharing enabled
ui.launch(share=True)

# Add ATS optimization analysis to the evaluation


# Function to extract keywords from a job description
def extract_job_description_keywords(job_description_text):
    """
    Extract key requirements and skills from a job description text
    to use for resume matching and evaluation.
    """
    if not job_description_text or len(job_description_text) < 50:
        return []
    
    # Process with spaCy for better entity recognition
    doc = nlp(job_description_text)
    
    # Extract potential skill keywords (nouns and noun phrases)
    keywords = []
    for chunk in doc.noun_chunks:
        if 3 <= len(chunk.text) <= 40:  # Reasonable length for a skill/requirement
            keywords.append(chunk.text.lower())
    
    # Extract technical terms and specific requirements
    tech_pattern = r'\b(?:proficient|experience|knowledge|skills?|familiarity|expertise)\s+(?:in|with|of)?\s+([A-Za-z0-9+#\s]{3,40}?)(?:\.|\band\b|,|\n|\))'
    tech_matches = re.findall(tech_pattern, job_description_text, re.IGNORECASE)
    keywords.extend([match.strip().lower() for match in tech_matches])
    
    # Extract years of experience requirements
    exp_pattern = r'(\d+[-\+]?\s*(?:years|yrs)(?:\s+of)?\s+(?:experience|exp)(?:\s+in)?\s+([A-Za-z0-9+#\s]{3,40}?)(?:\.|\band\b|,|\n|\)))'
    exp_matches = re.findall(exp_pattern, job_description_text, re.IGNORECASE)
    keywords.extend([match[0].strip().lower() for match in exp_matches])
    
    # Look for education requirements
    edu_pattern = r'\b(Bachelor|Master|MBA|PhD|BS|BA|MS|MA|Doctor|Associate)\b(?:[^\n.]*)'
    edu_matches = re.findall(edu_pattern, job_description_text, re.IGNORECASE)
    keywords.extend([match.strip().lower() for match in edu_matches])
    
    # Extract required certifications
    cert_pattern = r'\b(certifications?|certified|license[ds]?)\s+(?:in|as)?\s+([A-Za-z0-9+#\s]{3,40}?)(?:\.|\band\b|,|\n|\)|or)'
    cert_matches = re.findall(cert_pattern, job_description_text, re.IGNORECASE)
    keywords.extend([match[1].strip().lower() for match in cert_matches])
    
    # Clean up and remove duplicates
    clean_keywords = []
    for kw in keywords:
        # Basic cleanup and normalization
        kw = re.sub(r'[^\w\s\-\+#]', '', kw)  # Remove special chars except -, +, #
        kw = re.sub(r'\s+', ' ', kw).strip()  # Normalize whitespace
        
        if kw and len(kw) > 3 and kw not in clean_keywords and kw.lower() not in stopwords.words('english'):
            clean_keywords.append(kw)
    
    return clean_keywords[:50]  # Limit to top 50 keywords

# Function to match resume against job description
def analyze_resume_job_match(resume_text, job_description, job_title):
    """
    Analyze how well a resume matches a specific job description.
    Returns a match score and detailed analysis of matching and missing keywords.
    """
    # Extract keywords from job description
    job_keywords = extract_job_description_keywords(job_description)
    
    if not job_keywords:
        return {
            "match_score": 0,
            "matching_keywords": [],
            "missing_keywords": [],
            "has_job_description": False
        }
    
    # Convert resume text to lowercase for matching
    resume_lower = resume_text.lower()
    
    # Find matching keywords
    matching_keywords = []
    missing_keywords = []
    
    for keyword in job_keywords:
        keyword_lower = keyword.lower()
        # Try exact match first
        if keyword_lower in resume_lower:
            matching_keywords.append(keyword)
        # Try partial matches for longer phrases
        elif len(keyword) > 10:
            words = keyword_lower.split()
            # If most words in the phrase are present, count as partial match
            if sum(1 for word in words if word in resume_lower and len(word) > 3) > len(words) / 2:
                matching_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    
    # Calculate match score if job keywords exist
    if job_keywords:
        match_score = (len(matching_keywords) / len(job_keywords)) * 100
    else:
        match_score = 0
    
    # Limit the number of reported keywords for readability
    return {
        "match_score": round(match_score, 1),
        "matching_keywords": matching_keywords[:10],
        "missing_keywords": missing_keywords[:10],
        "has_job_description": True
    }

# Improve skill extraction to filter out irrelevant items


# Helper function to extract key sections from resume text
def extract_key_info(text):
    """Extract key information from resume text."""
    sections = {}
    
    # Extract summary/objective section
    summary_match = re.search(r'(summary|objective|profile).*?(?=experience|education|skills|$)', 
                            text, re.IGNORECASE | re.DOTALL)
    if summary_match:
        sections['summary'] = summary_match.group(0)[:300]
    
    # Extract experience section
    exp_match = re.search(r'(experience|work|employment).*?(?=education|skills|references|$)', 
                         text, re.IGNORECASE | re.DOTALL)
    if exp_match:
        sections['experience'] = exp_match.group(0)[:500]
    
    # Extract education section
    edu_match = re.search(r'(education|academic|degree).*?(?=experience|skills|references|$)', 
                          text, re.IGNORECASE | re.DOTALL)
    if edu_match:
        sections['education'] = edu_match.group(0)[:300]
    
    return sections





def analyze_section_with_ai(section_text, section_type, job_title, model=None, tokenizer=None):
    """
    Use AI to provide deeper analysis of resume sections.
    Returns detailed insights and recommendations.
    """
    analysis = []
    
    # Get AI-powered insights if model is available
    if model and tokenizer:
        ai_insights = get_ai_insights(section_text, f"{section_type} section for {job_title} role", model, tokenizer)
        analysis.extend(ai_insights)
    
    if not section_text:
        analysis.append("⚠ This section is empty or missing")
        return analysis
        
    if section_type == 'experience':
        # Analyze impact and achievements
        achievements = detect_achievements(section_text)
        if achievements:
            analysis.append("✓ Strong action verbs and quantifiable achievements detected")
        else:
            analysis.append("⚠ Consider adding more quantifiable achievements and metrics")
            
        # Check for relevant experience alignment
        if job_title.lower() in section_text.lower():
            analysis.append("✓ Experience aligns well with target role")
        else:
            analysis.append("⚠ Consider highlighting experience more relevant to " + job_title)
            
        # Analyze experience progression
        if len(section_text.split('\n')) > 5:
            analysis.append("✓ Shows clear career progression")
        
    elif section_type == 'education':
        # Check for relevant coursework
        if 'coursework' in section_text.lower():
            analysis.append("✓ Relevant coursework included")
        else:
            analysis.append("⚠ Consider adding relevant coursework if recent graduate")
            
        # Check for academic achievements
        academic_keywords = ['honors', 'distinction', 'gpa', 'dean', 'scholarship']
        if any(keyword in section_text.lower() for keyword in academic_keywords):
            analysis.append("✓ Academic achievements highlighted")
            
    elif section_type == 'skills':
        # Analyze skill categorization
        if '\n' in section_text:
            analysis.append("✓ Skills well-organized into categories")
        else:
            analysis.append("⚠ Consider organizing skills into clear categories")
            
        # Check for skill levels
        proficiency_keywords = ['proficient', 'experienced', 'familiar', 'expert']
        if any(keyword in section_text.lower() for keyword in proficiency_keywords):
            analysis.append("✓ Skill proficiency levels indicated")
        else:
            analysis.append("⚠ Consider indicating proficiency levels for key skills")
            
    elif section_type == 'summary':
        # Check for personal branding
        if job_title.lower() in section_text.lower():
            analysis.append("✓ Summary aligned with target role")
        else:
            analysis.append("⚠ Consider tailoring summary to target role")
            
        # Check for unique value proposition
        if 'years' in section_text.lower() or 'experience' in section_text.lower():
            analysis.append("✓ Experience level clearly stated")
            
    return analysis

def initialize_ai_model():
    """
    Initialize the Qwen2.5-1.5B-Instruct model for enhanced resume analysis.
    """
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16  # Use float16 for efficiency
        )
        return model, tokenizer
    except Exception as e:
        print(f"Warning: Could not initialize AI model: {e}")
        return None, None

def get_ai_insights(text, context, model, tokenizer):
    """
    Get AI-powered insights about the resume text.
    """
    if not model or not tokenizer:
        return []
        
    prompt = f"""Analyze this resume section in the context of {context}. 
    Provide specific, actionable feedback about strengths and areas for improvement.
    Focus on relevance, impact, and clarity.
    
    Resume text:
    {text}
    """
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Resume text:")[0].strip().split("\n")
    except Exception as e:
        print(f"Warning: AI analysis failed: {e}")
        return []

# Initialize the model when the script starts
try:
    model, tokenizer = initialize_ai_model()
except Exception as e:
    print(f"Warning: Could not initialize AI model. Running in basic mode. Error: {e}")
    model, tokenizer = None, None
