from tkinter import *
from tkinter import filedialog, messagebox
from PyPDF2 import PdfReader
from docx import Document
import joblib
import re
import os

# Load models
rf_classifier_categorization = joblib.load('rf_classifier_categorization.joblib')
tfidf_vectorizer_categorization = joblib.load('tfidf_vectorizer_categorization.joblib')
rf_classifier_job_recommendation = joblib.load('rf_classifier_job_recommendation.joblib')
tfidf_vectorizer_job_recommendation = joblib.load('tfidf_vectorizer_job_recommendation.joblib')

# Functions to process resumes
def cleanResume(txt):
    cleanText = re.sub(r"http\s+", " ", txt)
    cleanText = re.sub(r"RT|cc", " ", cleanText)
    cleanText = re.sub(r"#\s+", " ", cleanText)
    cleanText = re.sub(r"@\s+", " ", cleanText)
    cleanText = re.sub(r"[%s]" % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", cleanText)
    cleanText = re.sub(r"[^\x00-\x7f]", " ", cleanText)
    cleanText = re.sub(r"\s+", " ", cleanText)
    return cleanText

def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    return rf_classifier_categorization.predict(resume_tfidf)[0]

def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    return rf_classifier_job_recommendation.predict(resume_tfidf)[0]

def extract_contact_number_from_resume(text):
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

def extract_email_from_resume(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

def extract_skills_from_resume(text):
    skills_list = [
        'Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'Deep Learning', 'SQL',
        'Tableau',
        'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'React', 'Angular', 'Node.js', 'MongoDB', 'Express.js', 'Git',
        'Research', 'Statistics', 'Quantitative Analysis', 'Qualitative Analysis', 'SPSS', 'R', 'Data Visualization',
        'Matplotlib',
        'Seaborn', 'Plotly', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'NLTK', 'Text Mining',
        'Natural Language Processing', 'Computer Vision', 'Image Processing', 'OCR', 'Speech Recognition',
        'Recommendation Systems',
        'Collaborative Filtering', 'Content-Based Filtering', 'Reinforcement Learning', 'Neural Networks',
        'Convolutional Neural Networks',
        'Recurrent Neural Networks', 'Generative Adversarial Networks', 'XGBoost', 'Random Forest', 'Decision Trees',
        'Support Vector Machines',
        'Linear Regression', 'Logistic Regression', 'K-Means Clustering', 'Hierarchical Clustering', 'DBSCAN',
        'Association Rule Learning',
        'Apache Hadoop', 'Apache Spark', 'MapReduce', 'Hive', 'HBase', 'Apache Kafka', 'Data Warehousing', 'ETL',
        'Big Data Analytics',
        'Cloud Computing', 'Amazon Web Services (AWS)', 'Microsoft Azure', 'Google Cloud Platform (GCP)', 'Docker',
        'Kubernetes', 'Linux',
        'Shell Scripting', 'Cybersecurity', 'Network Security', 'Penetration Testing', 'Firewalls', 'Encryption',
        'Malware Analysis',
        'Digital Forensics', 'CI/CD', 'DevOps', 'Agile Methodology', 'Scrum', 'Kanban', 'Continuous Integration',
        'Continuous Deployment',
        'Software Development', 'Web Development', 'Mobile Development', 'Backend Development', 'Frontend Development',
        'Full-Stack Development',
        'UI/UX Design', 'Responsive Design', 'Wireframing', 'Prototyping', 'User Testing', 'Adobe Creative Suite',
        'Photoshop', 'Illustrator',
        'InDesign', 'Figma', 'Sketch', 'Zeplin', 'InVision', 'Product Management', 'Market Research',
        'Customer Development', 'Lean Startup',
        'Business Development', 'Sales', 'Marketing', 'Content Marketing', 'Social Media Marketing', 'Email Marketing',
        'SEO', 'SEM', 'PPC',
        'Google Analytics', 'Facebook Ads', 'LinkedIn Ads', 'Lead Generation', 'Customer Relationship Management (CRM)',
        'Salesforce',
        'HubSpot', 'Zendesk', 'Intercom', 'Customer Support', 'Technical Support', 'Troubleshooting',
        'Ticketing Systems', 'ServiceNow',
        'ITIL', 'Quality Assurance', 'Manual Testing', 'Automated Testing', 'Selenium', 'JUnit', 'Load Testing',
        'Performance Testing',
        'Regression Testing', 'Black Box Testing', 'White Box Testing', 'API Testing', 'Mobile Testing',
        'Usability Testing', 'Accessibility Testing',
        'Cross-Browser Testing', 'Agile Testing', 'User Acceptance Testing', 'Software Documentation',
        'Technical Writing', 'Copywriting',
        'Editing', 'Proofreading', 'Content Management Systems (CMS)', 'WordPress', 'Joomla', 'Drupal', 'Magento',
        'Shopify', 'E-commerce',
        'Payment Gateways', 'Inventory Management', 'Supply Chain Management', 'Logistics', 'Procurement',
        'ERP Systems', 'SAP', 'Oracle',
        'Microsoft Dynamics', 'Tableau', 'Power BI', 'QlikView', 'Looker', 'Data Warehousing', 'ETL',
        'Data Engineering', 'Data Governance',
        'Data Quality', 'Master Data Management', 'Predictive Analytics', 'Prescriptive Analytics',
        'Descriptive Analytics', 'Business Intelligence',
        'Dashboarding', 'Reporting', 'Data Mining', 'Web Scraping', 'API Integration', 'RESTful APIs', 'GraphQL',
        'SOAP', 'Microservices',
        'Serverless Architecture', 'Lambda Functions', 'Event-Driven Architecture', 'Message Queues', 'GraphQL',
        'Socket.io', 'WebSockets'
                     'Ruby', 'Ruby on Rails', 'PHP', 'Symfony', 'Laravel', 'CakePHP', 'Zend Framework', 'ASP.NET', 'C#',
        'VB.NET', 'ASP.NET MVC', 'Entity Framework',
        'Spring', 'Hibernate', 'Struts', 'Kotlin', 'Swift', 'Objective-C', 'iOS Development', 'Android Development',
        'Flutter', 'React Native', 'Ionic',
        'Mobile UI/UX Design', 'Material Design', 'SwiftUI', 'RxJava', 'RxSwift', 'Django', 'Flask', 'FastAPI',
        'Falcon', 'Tornado', 'WebSockets',
        'GraphQL', 'RESTful Web Services', 'SOAP', 'Microservices Architecture', 'Serverless Computing', 'AWS Lambda',
        'Google Cloud Functions',
        'Azure Functions', 'Server Administration', 'System Administration', 'Network Administration',
        'Database Administration', 'MySQL', 'PostgreSQL',
        'SQLite', 'Microsoft SQL Server', 'Oracle Database', 'NoSQL', 'MongoDB', 'Cassandra', 'Redis', 'Elasticsearch',
        'Firebase', 'Google Analytics',
        'Google Tag Manager', 'Adobe Analytics', 'Marketing Automation', 'Customer Data Platforms', 'Segment',
        'Salesforce Marketing Cloud', 'HubSpot CRM',
        'Zapier', 'IFTTT', 'Workflow Automation', 'Robotic Process Automation (RPA)', 'UI Automation',
        'Natural Language Generation (NLG)',
        'Virtual Reality (VR)', 'Augmented Reality (AR)', 'Mixed Reality (MR)', 'Unity', 'Unreal Engine', '3D Modeling',
        'Animation', 'Motion Graphics',
        'Game Design', 'Game Development', 'Level Design', 'Unity3D', 'Unreal Engine 4', 'Blender', 'Maya',
        'Adobe After Effects', 'Adobe Premiere Pro',
        'Final Cut Pro', 'Video Editing', 'Audio Editing', 'Sound Design', 'Music Production', 'Digital Marketing',
        'Content Strategy', 'Conversion Rate Optimization (CRO)',
        'A/B Testing', 'Customer Experience (CX)', 'User Experience (UX)', 'User Interface (UI)', 'Persona Development',
        'User Journey Mapping', 'Information Architecture (IA)',
        'Wireframing', 'Prototyping', 'Usability Testing', 'Accessibility Compliance', 'Internationalization (I18n)',
        'Localization (L10n)', 'Voice User Interface (VUI)',
        'Chatbots', 'Natural Language Understanding (NLU)', 'Speech Synthesis', 'Emotion Detection',
        'Sentiment Analysis', 'Image Recognition', 'Object Detection',
        'Facial Recognition', 'Gesture Recognition', 'Document Recognition', 'Fraud Detection',
        'Cyber Threat Intelligence', 'Security Information and Event Management (SIEM)',
        'Vulnerability Assessment', 'Incident Response', 'Forensic Analysis', 'Security Operations Center (SOC)',
        'Identity and Access Management (IAM)', 'Single Sign-On (SSO)',
        'Multi-Factor Authentication (MFA)', 'Blockchain', 'Cryptocurrency', 'Decentralized Finance (DeFi)',
        'Smart Contracts', 'Web3', 'Non-Fungible Tokens (NFTs)']
    
    skills = []

    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)

    return skills

def extract_education_from_resume(text):
    education = []

    # List of education keywords to match against
    education_keywords = [
        'Computer Science', 'Information Technology', 'Software Engineering', 'Electrical Engineering', 'Mechanical Engineering', 'Civil Engineering',
        'Chemical Engineering', 'Biomedical Engineering', 'Aerospace Engineering', 'Nuclear Engineering', 'Industrial Engineering', 'Systems Engineering',
        'Environmental Engineering', 'Petroleum Engineering', 'Geological Engineering', 'Marine Engineering', 'Robotics Engineering', 'Biotechnology',
        'Biochemistry', 'Microbiology', 'Genetics', 'Molecular Biology', 'Bioinformatics', 'Neuroscience', 'Biophysics', 'Biostatistics', 'Pharmacology',
        'Physiology', 'Anatomy', 'Pathology', 'Immunology', 'Epidemiology', 'Public Health', 'Health Administration', 'Nursing', 'Medicine', 'Dentistry',
        'Pharmacy', 'Veterinary Medicine', 'Medical Technology', 'Radiography', 'Physical Therapy', 'Occupational Therapy', 'Speech Therapy', 'Nutrition',
        'Sports Science', 'Kinesiology', 'Exercise Physiology', 'Sports Medicine', 'Rehabilitation Science', 'Psychology', 'Counseling', 'Social Work',
        'Sociology', 'Anthropology', 'Criminal Justice', 'Political Science', 'International Relations', 'Economics', 'Finance', 'Accounting', 'Business Administration',
        'Management', 'Marketing', 'Entrepreneurship', 'Hospitality Management', 'Tourism Management', 'Supply Chain Management', 'Logistics Management',
        'Operations Management', 'Human Resource Management', 'Organizational Behavior', 'Project Management', 'Quality Management', 'Risk Management',
        'Strategic Management', 'Public Administration', 'Urban Planning', 'Architecture', 'Interior Design', 'Landscape Architecture', 'Fine Arts',
        'Visual Arts', 'Graphic Design', 'Fashion Design', 'Industrial Design', 'Product Design', 'Animation', 'Film Studies', 'Media Studies',
        'Communication Studies', 'Journalism', 'Broadcasting', 'Creative Writing', 'English Literature', 'Linguistics', 'Translation Studies',
        'Foreign Languages', 'Modern Languages', 'Classical Studies', 'History', 'Archaeology', 'Philosophy', 'Theology', 'Religious Studies',
        'Ethics', 'Education', 'Early Childhood Education', 'Elementary Education', 'Secondary Education', 'Special Education', 'Higher Education',
        'Adult Education', 'Distance Education', 'Online Education', 'Instructional Design', 'Curriculum Development'
        'Library Science', 'Information Science', 'Computer Engineering', 'Software Development', 'Cybersecurity', 'Information Security',
        'Network Engineering', 'Data Science', 'Data Analytics', 'Business Analytics', 'Operations Research', 'Decision Sciences',
        'Human-Computer Interaction', 'User Experience Design', 'User Interface Design', 'Digital Marketing', 'Content Strategy',
        'Brand Management', 'Public Relations', 'Corporate Communications', 'Media Production', 'Digital Media', 'Web Development',
        'Mobile App Development', 'Game Development', 'Virtual Reality', 'Augmented Reality', 'Blockchain Technology', 'Cryptocurrency',
        'Digital Forensics', 'Forensic Science', 'Criminalistics', 'Crime Scene Investigation', 'Emergency Management', 'Fire Science',
        'Environmental Science', 'Climate Science', 'Meteorology', 'Geography', 'Geomatics', 'Remote Sensing', 'Geoinformatics',
        'Cartography', 'GIS (Geographic Information Systems)', 'Environmental Management', 'Sustainability Studies', 'Renewable Energy',
        'Green Technology', 'Ecology', 'Conservation Biology', 'Wildlife Biology', 'Zoology']

    for keyword in education_keywords:
        pattern = r"(?i)\b{}\b".format(re.escape(keyword))
        match = re.search(pattern, text)
        if match:
            education.append(match.group())

    return education     

def extract_name_from_resume(text):
    pattern = r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)"
    match = re.search(pattern, text)
    return match.group() if match else None

def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

def read_docx(filepath):
    doc = Document(filepath)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return '\n'.join(text)

def upload_resume():
    global resume_text
    filepath = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), 
                                                     ("Text files", "*.txt"),
                                                     ("Word documents", "*.docx")])
    
    if not filepath:
        return  # No file selected
    
    # Read file based on extension
    if filepath.endswith('.pdf'):
        text = pdf_to_text(filepath)
    elif filepath.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    elif filepath.endswith('.docx'):
        text = read_docx(filepath)
    else:
        messagebox.showerror("Error", "Invalid file format. Please upload a PDF, TXT, or DOCX file.")
        return
    
    # Display only the file name in the upload label
    filename = os.path.basename(filepath)
    upload.config(text=filename)
    upload.update()

    # Store the text globally to use it in the check function
    resume_text = text

    upload_btn.config(text="Check", command=check_resume)


def check_resume():
    if not resume_text:
        messagebox.showwarning("No File", "Please upload a resume first.")
        return

    # Create a new Toplevel window
    result_window = Toplevel(root)
    result_window.title("Resume Analysis")
    result_window.geometry("700x500+80+120")
    result_window.config(bg="#055B7F")

    # Create and place labels for the results
    results_frame = Frame(result_window, bg="#013a5c")
    results_frame.pack(padx=20, pady=20, fill=BOTH, expand=True)

    category = predict_category(resume_text)
    job_recommend = job_recommendation(resume_text)
    contact_number = extract_contact_number_from_resume(resume_text)
    email = extract_email_from_resume(resume_text)
    skills = extract_skills_from_resume(resume_text)
    name = extract_name_from_resume(resume_text)
    education = extract_education_from_resume(resume_text)

    result_text = (
        f"Name: {name}\n"
        f"\nContact Number: {contact_number}\n"
        f"\nEmail: {email}\n"
        f"\nEducation: {education}\n"
        f"\nSkills: {', '.join(skills)}\n"
        f"\nCategory: {category}\n"
        f"\nJob Recommendation: {job_recommend}\n"    
    )

    result_lbl = Label(results_frame, text=result_text, font=("Roboto", 13),fg = "white", bg="#013a5c", anchor='w', justify=LEFT)
    result_lbl.pack(padx=10, pady=30)
    upload_btn.config(text = "Upload", command = upload_resume)
    

root = Tk()
root.geometry("1000x750+50+0")
root.title("Resume Screening AI-based System")
root.config(bg="#055B7F")

# Heading and Descriptions
window_frame = Frame(root, bg="#013a5c",bd = 3)
window_frame.place(x=120, y=50, width=760, height=650)

heading_lbl = Label(window_frame, text="Resume Screening AI-based System", font=("Roboto", 30, "bold"), fg="white", bg="#013a5c",justify = LEFT)
heading_lbl.grid(row=0, column=0, columnspan=2, padx=10, pady=20)

description_lbl = Label(window_frame, text="This system supports TXT and PDF files to be uploaded and it will work on the following:", font=("Roboto", 11), fg="white", bg="#013a5c")
description_lbl.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Feature List
feature_lbl1 = Label(window_frame, text="1- Resume Categorization", font=("Roboto", 11), fg="white", bg="#013a5c")
feature_lbl1.grid(row=2, column=0, sticky=W, padx=10, pady=10)

feature_lbl2 = Label(window_frame, text="2- Resume Job Recommendation", font=("Roboto", 11), fg="white", bg="#013a5c")
feature_lbl2.grid(row=3, column=0, sticky=W, padx=10, pady=10)

feature_lbl3 = Label(window_frame, text="3- Resume Parsing (Information Extraction)", font=("Roboto", 11), fg="white", bg="#013a5c")
feature_lbl3.grid(row=4, column=0, sticky=W, padx=10, pady=10)

# Upload Your Resume Label
upload_lbl = Label(window_frame, text="Upload Your Resume", font=("Roboto", 20, "bold"), fg="white", bg="#013a5c")
upload_lbl.grid(row=5, column=0, sticky=W, padx=230, pady=20)

# frame1 = Frame(window_frame, width=660, height=220, bg="#013a5c", relief="groove")
# frame1.grid(row=6, column=0, sticky=W, padx=50, pady=20)

upload = Label(window_frame,text = "..................................", font=("Roboto", 20, "bold"), fg="white", bg="#013a5c")
upload.grid(row=6, column=0, sticky=W, padx=230, pady=20)

# Button 
upload_btn = Button(window_frame, text="Upload", font=("Roboto", 20, "bold"), fg="white", bg="green", command=upload_resume)
upload_btn.grid(row=7, column=0, sticky=W, padx=300, pady=20)

root.mainloop()
