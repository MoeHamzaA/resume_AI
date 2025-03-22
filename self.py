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
    # Check for concise language
    sentences = nltk.sent_tokenize(text)
    avg_words_per_sentence = word_count / max(1, len(sentences))
    
    if avg_words_per_sentence > 25:
        metrics["clarity_score"] -= 1
        metrics["improvements"].append("Your sentences are quite long (avg. {:.1f} words). Use more concise language for better readability.".format(avg_words_per_sentence))
    elif avg_words_per_sentence < 10:
        metrics["clarity_score"] += 1
        metrics["strengths"].append("Good use of concise language throughout your resume.")
    
    # Check for passive voice (basic implementation)
    passive_count = len(re.findall(r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', text_lower))
    if passive_count > 5:
        metrics["clarity_score"] -= 1
        metrics["improvements"].append("Consider replacing passive voice with active voice for stronger impact.")
    
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
        'engineering': ['engineer', 'engineering', 'developer', 'programmer', 'architect', 'coder'],
        'management': ['manager', 'director', 'supervisor', 'lead', 'chief', 'head'],
        'analysis': ['analyst', 'researcher', 'scientist', 'specialist', 'consultant'],
        'design': ['designer', 'architect', 'ui/ux', 'ux', 'ui', 'graphic'],
        'support': ['support', 'assistant', 'help', 'technician', 'service'],
        'sales': ['sales', 'account', 'representative', 'business development', 'client'],
        'marketing': ['marketing', 'brand', 'seo', 'content', 'social media', 'growth'],
        'finance': ['finance', 'accounting', 'financial', 'accountant', 'bookkeeper', 'controller'],
        'hr': ['hr', 'human resources', 'recruiter', 'talent', 'people', 'recruitment'],
        'healthcare': ['doctor', 'nurse', 'physician', 'medical', 'clinical', 'health'],
        'education': ['teacher', 'professor', 'instructor', 'educator', 'tutor', 'trainer'],
        'legal': ['lawyer', 'attorney', 'legal', 'counsel', 'paralegal', 'compliance'],
        'creative': ['writer', 'editor', 'artist', 'creative', 'content', 'producer'],
        'customer': ['customer', 'client', 'service', 'success', 'support', 'experience'],
        'administrative': ['admin', 'administrative', 'coordinator', 'secretary', 'clerk', 'receptionist'],
        'operations': ['operations', 'logistic', 'supply chain', 'warehouse', 'inventory', 'production']
    }
    
    # Domain-specific dictionaries for popular fields
    domain_keywords = {
        'software': ['programming', 'coding', 'development', 'algorithms', 'testing', 'debugging', 
                    'software architecture', 'apis', 'web', 'mobile', 'frontend', 'backend', 'full stack',
                    'object-oriented', 'functional programming', 'version control', 'ci/cd'],
        
        'data': ['analytics', 'big data', 'data mining', 'statistics', 'data visualization', 
                'machine learning', 'artificial intelligence', 'deep learning', 'predictive modeling',
                'data warehousing', 'etl', 'bi', 'reporting', 'dashboards'],
        
        'cloud': ['aws', 'azure', 'gcp', 'infrastructure', 'virtualization', 'containers',
                 'microservices', 'serverless', 'iaas', 'paas', 'saas', 'cloud security',
                 'cloud architecture', 'devops', 'automation', 'scalability'],
        
        'security': ['cybersecurity', 'infosec', 'security protocols', 'penetration testing', 
                    'vulnerability assessment', 'encryption', 'authentication', 'compliance',
                    'security audits', 'threat detection', 'incident response'],
        
        'project': ['project management', 'agile', 'scrum', 'kanban', 'waterfall', 'prince2',
                   'risk management', 'requirements gathering', 'stakeholder management', 
                   'scheduling', 'budget management', 'resource allocation'],
        
        'finance': ['financial analysis', 'accounting', 'budgeting', 'forecasting', 'financial reporting',
                   'financial modeling', 'tax', 'audit', 'compliance', 'risk assessment',
                   'investment analysis', 'portfolio management'],
        
        'marketing': ['market research', 'branding', 'advertising', 'digital marketing', 'seo', 'sem', 
                     'social media marketing', 'content marketing', 'email marketing', 'campaign management',
                     'marketing analytics', 'conversion optimization'],
        
        'healthcare': ['patient care', 'clinical procedures', 'medical terminology', 'healthcare regulations',
                      'electronic health records', 'patient management', 'medical coding',
                      'treatment planning', 'disease management'],
        
        'hr': ['recruitment', 'talent acquisition', 'onboarding', 'employee relations', 'performance management',
              'compensation', 'benefits', 'hr policies', 'organizational development',
              'training', 'employee engagement', 'diversity and inclusion'],
        
        'legal': ['legal research', 'contracts', 'compliance', 'litigation', 'negotiation',
                 'legal documentation', 'case management', 'legal advice', 'legal analysis',
                 'regulatory affairs', 'intellectual property']
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
    
    # Add domain-specific feedback
    if job_analysis['domains']:
        main_domain = job_analysis['domains'][0] if job_analysis['domains'] else None
        
        if main_domain == 'software':
            feedback.append("Highlight specific programming languages and frameworks you're proficient in.")
            
        elif main_domain == 'data':
            feedback.append("Showcase your experience with data analysis tools and techniques.")
            
        elif main_domain == 'cloud':
            feedback.append("Demonstrate your experience with specific cloud platforms and services.")
            
        elif main_domain == 'security':
            feedback.append("Emphasize your knowledge of security protocols and compliance standards.")
    
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
        'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'php', 'html', 'css',
        'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'spring', 'rails',
        'aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes', 'terraform', 'ansible',
        'jenkins', 'git', 'ci/cd', 'devops', 'linux', 'unix', 'windows', 'bash', 'powershell',
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'nosql', 'database',
        'machine learning', 'ai', 'data science', 'analytics', 'tableau', 'power bi',
        'hadoop', 'spark', 'kafka', 'elasticsearch', 'networking', 'security', 'api',
        'rest', 'soap', 'json', 'xml', 'graphql', 'agile', 'scrum', 'kanban', 'waterfall',
        'jira', 'confluence', 'slack', 'teams', 'office', 'excel', 'word', 'powerpoint',
        'photoshop', 'illustrator', 'figma', 'sketch', 'indesign', 'autocad',
        'leadership', 'project management', 'communication', 'teamwork', 'problem solving',
        'critical thinking', 'time management', 'organization', 'presentation', 'negotiation',
        'cybersecurity', 'penetration testing', 'vulnerability', 'compliance', 'auditing',
        'iot', 'blockchain', 'virtualization', 'containerization', 'serverless',
        'ec2', 's3', 'lambda', 'route 53', 'vpc', 'iam', 'rds', 'dynamodb',
        'cloud security', 'network security', 'firewall', 'encryption', 'authentication',
        # Add non-tech professional skills
        'project management', 'leadership', 'team management', 'budgeting', 'forecasting',
        'strategic planning', 'business development', 'sales', 'marketing', 'customer service',
        'account management', 'client relationship', 'operations management', 'supply chain',
        'logistics', 'quality assurance', 'public speaking', 'writing', 'editing',
        'research', 'analysis', 'reporting', 'training', 'mentoring', 'coaching',
        'negotiation', 'conflict resolution', 'decision making', 'problem solving',
        'critical thinking', 'creativity', 'innovation', 'adaptability', 'flexibility',
        # Healthcare skills
        'patient care', 'clinical', 'medical terminology', 'electronic health records',
        'treatment planning', 'diagnostics', 'patient assessment', 'vital signs',
        'medical coding', 'medical billing', 'pharmacy', 'pharmacology',
        # Finance/accounting skills
        'accounting', 'bookkeeping', 'financial analysis', 'financial reporting',
        'tax preparation', 'budgeting', 'forecasting', 'financial modeling',
        'quickbooks', 'sap', 'oracle financials', 'investment analysis',
        # Educational skills
        'curriculum development', 'lesson planning', 'student assessment',
        'classroom management', 'educational technology', 'instructional design',
        'teaching', 'training', 'e-learning', 'remote teaching'
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
        result += "\n#### Work Experience Analysis:\n"
        exp_length = len(resume_sections['experience'].split())
        if exp_length > 300:
            result += "- Your experience section is well-developed with sufficient detail.\n"
            # Extract job titles/positions mentioned
            positions = re.findall(r'(?:^|\n)([A-Z][a-zA-Z\s]+)(?=\s*\b(?:at|,|\-|\(|from|to)\b)', resume_sections['experience'])
            if positions:
                result += f"- Positions identified: {', '.join(positions[:3])}" + (f" and {len(positions)-3} more" if len(positions) > 3 else "") + "\n"
        else:
            result += "- Consider expanding your work experience with more specific achievements and responsibilities.\n"
    
    if 'education' in resume_sections:
        result += "\n#### Education Analysis:\n"
        # Extract education details
        degrees = re.findall(r'\b(?:Bachelor|Master|MBA|PhD|BS|BA|MS|MA|Doctor|Associate)\b(?:[^\n.]*)', resume_sections['education'])
        if degrees:
            result += f"- Education credentials identified: {', '.join(degrees[:2])}" + (f" and more" if len(degrees) > 2 else "") + "\n"
        else:
            result += "- Consider formatting your education section to clearly highlight degrees and institutions.\n"
        
        if is_student:
            result += "- **Student Status Detected:** As a student, focus on relevant coursework, projects, and technical skills to compensate for limited work experience.\n"
    
    # Add skill analysis with improved extraction
    if 'skills' in resume_sections:
        result += "\n#### Skills Analysis:\n"
        # Extract skills mentioned using our improved function
        skill_text = resume_sections['skills']
        skill_length = len(skill_text.split())
        
        if skill_length > 50:
            result += "- Comprehensive skills section with good detail.\n"
        else:
            result += "- Consider expanding your skills section with more specific technical and soft skills.\n"
        
        # Use improved skill extraction
        skills_list = extract_skills_from_text(skill_text)
        if skills_list:
            result += f"- **Key skills identified:** {', '.join(skills_list[:5])}" + (f" and {len(skills_list)-5} more" if len(skills_list) > 5 else "") + "\n"
            
        # Add role-specific skill gap analysis based on job title analysis
        if job_analysis['skills']:
            # Match job-specific skills the resume mentions
            role_specific_skills = job_analysis['skills']
            found_role_skills = [skill for skill in role_specific_skills if any(skill.lower() in s.lower() for s in skills_list)]
            missing_role_skills = [skill for skill in role_specific_skills if not any(skill.lower() in s.lower() for s in skills_list)]
            
            if found_role_skills:
                result += f"- **{job_title}-relevant skills identified:** {', '.join(found_role_skills[:5])}\n"
            
            if missing_role_skills and len(missing_role_skills) > 2:
                result += f"- **Consider adding these {job_title}-relevant skills if you have experience with them:** {', '.join(missing_role_skills[:5])}\n"
    
    # Add job-specific recommendations based on job analysis
    result += f"\n#### {job_title}-Specific Recommendations:\n"
    
    # Get job-specific recommendations from our function
    role_recommendations = job_recommendations[:4]  # Take top 4 recommendations
    
    for recommendation in role_recommendations:
        result += f"- {recommendation}\n"
        
    # Add seniority-specific advice
    if job_analysis['seniority'] == 'junior':
        result += "- For entry-level positions, emphasize your education, relevant coursework, and willingness to learn\n"
    elif job_analysis['seniority'] == 'senior':
        result += "- For senior positions, emphasize leadership experience and strategic contributions\n"
    elif job_analysis['seniority'] == 'manager':
        result += "- For management positions, highlight team leadership, budget management, and strategic planning\n"
    
    # Add strategic recommendations
    result += "\n#### Strategic Recommendations:\n"
    
    strategic_recommendations = job_recommendations[4:] if len(job_recommendations) > 4 else ["Tailor your resume for each application", "Quantify achievements whenever possible"]
    
    for i, recommendation in enumerate(strategic_recommendations[:3]):
        result += f"- **Tip {i+1}:** {recommendation}\n"
    
    # Add resume structure analysis
    result += "\n#### Resume Structure Analysis:\n"
    missing_sections = []
    for section in ['summary', 'experience', 'education', 'skills', 'projects']:
        if section not in resume_sections:
            # For students, don't penalize missing experience as heavily
            if section == 'experience' and is_student:
                continue
            missing_sections.append(section)
    
    if missing_sections:
        result += f"- Consider adding these missing sections: {', '.join(missing_sections)}.\n"
    else:
        result += "- All essential resume sections are present.\n"
    
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
    edu_pattern = r'\b(Bachelor|Master|MBA|PhD|BS|BA|MS|MA|Doctorate|Associate)(?:\'s)?\s+(?:degree)?\s+(?:in)?\s+([A-Za-z\s]{3,40}?)(?:\.|\band\b|,|\n|\)|or)'
    edu_matches = re.findall(edu_pattern, job_description_text, re.IGNORECASE)
    keywords.extend([f"{match[0]} in {match[1]}".strip().lower() for match in edu_matches])
    
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
    
    # Calculate match score (0-100%)
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
    """
    Extract key sections from resume text for additional analysis
    """
    sections = {}
    # Attempt to extract experience section
    exp_match = re.search(r'(experience|work|employment).*?(?=education|skills|references|$)', 
                          text, re.IGNORECASE | re.DOTALL)
    if exp_match:
        sections['experience'] = exp_match.group(0)[:500]
    
    # Extract skills section
    skills_match = re.search(r'(skills|technologies|proficiencies).*?(?=experience|education|references|$)', 
                             text, re.IGNORECASE | re.DOTALL)
    if skills_match:
        sections['skills'] = skills_match.group(0)[:300]
    
    # Extract education section
    edu_match = re.search(r'(education|academic|degree).*?(?=experience|skills|references|$)', 
                          text, re.IGNORECASE | re.DOTALL)
    if edu_match:
        sections['education'] = edu_match.group(0)[:300]
    
    return sections
