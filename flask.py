from flask import Flask, request, jsonify
import pdfplumber
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('wordnet')

app = Flask(__name__)

# Preprocessing function for text
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Function to calculate matching score
def calculate_matching_score(job_text, resume_text):
    # Preprocess job and resume text
    job_text = preprocess_text(job_text)
    resume_text = preprocess_text(resume_text)

    # Vectorize job and resume text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([job_text, resume_text])

    # Calculate cosine similarity
    similarity_score = cosine_similarity(X[0], X[1])[0][0]
    matching_score = similarity_score * 100
    return matching_score

@app.route('/match-resume', methods=['POST'])
def match_resume():
    # Check if files are in the request
    if 'job_circular' not in request.files or 'resume' not in request.files:
        return jsonify({'error': 'Job circular and resume files are required'}), 400

    # Get job circular and resume files
    job_circular_file = request.files['job_circular']
    resume_file = request.files['resume']

    # Extract text from job circular
    with pdfplumber.open(job_circular_file) as pdf:
        job_text = ''
        for page in pdf.pages:
            job_text += page.extract_text()

    # Extract text from resume
    with pdfplumber.open(resume_file) as pdf:
        resume_text = ''
        for page in pdf.pages:
            resume_text += page.extract_text()

    print(resume_text)
    # Calculate matching score
    matching_score = calculate_matching_score(job_text, resume_text)
    print (matching_score)

    return jsonify({'matching_score': matching_score}), 200

if __name__ == '__main__':
    app.run(debug=True)
