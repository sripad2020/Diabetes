from werkzeug.utils import secure_filename
import PyPDF2
import logging
import sqlite3
import os
import re,json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import google.generativeai as genai

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Do_you_know_me?'
DB_PATH = 'users.db'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyC9lsET5jCJJOZmoPQ8k8TeMqeYvTvhIfk')  # Fallback to provided key
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not set in environment variables.")
genai.configure(api_key=GOOGLE_API_KEY)

# Load machine learning models
try:
    XGB_MODEL = joblib.load('diabetes_5050_XGB.pkl')
    EXTRA_TREES_MODEL = joblib.load('diabetes_5050_ExtraTrees.pkl')
except FileNotFoundError:
    print("Error: Model files not found. Ensure 'diabetes_5050_XGB.pkl' and 'diabetes_5050_ExtraTrees.pkl' are in the project root.")
    XGB_MODEL = None
    EXTRA_TREES_MODEL = None

FEATURES = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'HeartDiseaseorAttack',
            'AnyHealthcare', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age']

def clean_text(text):
    return re.sub(r'\*\*|\*', '', text)

def clean_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'`{3}.*?`{3}', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'^\s*>+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\*\-+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

if not os.path.exists(DB_PATH):
    init_db()

def generate_gemini_analysis(prediction, confidence, input_data):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        input_description = "\n".join([f"{feature}: {value}" for feature, value in input_data.items()])
        prompt = f"""
        You are a medical assistant analyzing a diabetes risk prediction from a machine learning model. The model predicted:
        - Result: {'Diabetes Risk Detected' if prediction == 1 else 'No Diabetes Risk'}
        - Confidence: {confidence:.1f}%
        Input features:
        {input_description}
        Provide a clear, concise analysis (150-200 words) as a bulleted list with exactly 5 points:
        - Interpret the prediction and its implications for the user.
        - Identify 1-2 key input factors likely influencing the prediction.
        - Explain the confidence level and its reliability.
        - Suggest specific next steps or lifestyle changes.
        - Highlight limitations of the prediction.
        Use a professional, empathetic tone suitable for a non-medical audience.
        """
        response = model.generate_content(prompt)
        cleaned_response = clean_markdown(clean_text(response.text))
        return cleaned_response
    except Exception as e:
        print(f"Error in Gemini API: {str(e)}")
        return "Unable to generate analysis due to an error. Please consult a healthcare professional for detailed advice."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, password FROM users WHERE email = ?', (email,))
            user = cursor.fetchone()
            if user and check_password_hash(user[1], password):
                session['user_id'] = user[0]
                flash('Login successful!', 'success')
                return redirect('/predict')
            else:
                flash('Invalid email or password', 'error')
                return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form.get('confirm-password')
        if not name or not email or not password or not confirm_password:
            flash('All fields are required', 'error')
            return redirect(url_for('signup'))
        if len(name.strip()) < 1:
            flash('Please enter a valid full name', 'error')
            return redirect(url_for('signup'))
        if not re.match(r'^[\w\-\.]+@([\w\-]+\.)+[\w\-]{2,4}$', email):
            flash('Please enter a valid email address', 'error')
            return redirect(url_for('signup'))
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return redirect(url_for('signup'))
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
                cursor.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                              (name, email, hashed_password))
                conn.commit()
                flash('Account created successfully! Please log in.', 'success')
                return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists', 'error')
            return redirect(url_for('signup'))
    return render_template('signup.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        flash('Please log in to make a prediction', 'error')
        return redirect(url_for('login'))
    prediction_result = None
    form_data = None
    if request.method == 'POST':
        try:
            print("Form data:", request.form)
            form_data = {}
            for feature in FEATURES:
                value = request.form.get(feature)
                if not value:
                    flash(f'Missing or empty value for {feature}. Please select an option.', 'error')
                    return redirect(url_for('predict'))
                try:
                    value = float(value)
                except ValueError:
                    flash(f'Invalid numeric value for {feature}. Please enter a valid number.', 'error')
                    return redirect(url_for('predict'))
                if feature == 'BMI' and (value < 12 or value > 98):
                    flash('BMI must be between 12 and 98.', 'error')
                    return redirect(url_for('predict'))
                elif feature in ['MentHlth', 'PhysHlth'] and (value < 0 or value > 30):
                    flash(f'{feature} must be between 0 and 30 days.', 'error')
                    return redirect(url_for('predict'))
                elif feature == 'GenHlth' and (value < 1 or value > 5):
                    flash('General Health must be between 1 and 5.', 'error')
                    return redirect(url_for('predict'))
                elif feature == 'Age' and (value < 1 or value > 13):
                    flash('Age category must be between 1 and 13.', 'error')
                    return redirect(url_for('predict'))
                elif feature not in ['BMI', 'MentHlth', 'PhysHlth', 'GenHlth', 'Age'] and value not in [0, 1]:
                    flash(f'{feature} must be 0 or 1.', 'error')
                    return redirect(url_for('predict'))
                form_data[feature] = value
            input_data = np.array([form_data[feature] for feature in FEATURES]).reshape(1, -1)
            if XGB_MODEL is None or EXTRA_TREES_MODEL is None:
                flash('Prediction models are not available. Please contact support.', 'error')
                return redirect(url_for('predict'))
            # Perform predictions and validate outputs
            try:
                xgb_pred = XGB_MODEL.predict(input_data)[0]
                extra_trees_pred = EXTRA_TREES_MODEL.predict(input_data)[0]
                xgb_prob = XGB_MODEL.predict_proba(input_data)[0][1] * 100 if hasattr(XGB_MODEL, 'predict_proba') else 50.0
                extra_trees_prob = EXTRA_TREES_MODEL.predict_proba(input_data)[0][1] * 100 if hasattr(EXTRA_TREES_MODEL, 'predict_proba') else 50.0
            except Exception as e:
                print(f"Model prediction error: {str(e)}")
                flash('Error in model prediction. Please try again or contact support.', 'error')
                return redirect(url_for('predict'))
            # Ensure probabilities are valid
            if not (0 <= xgb_prob <= 100 and 0 <= extra_trees_prob <= 100):
                print(f"Invalid probabilities: xgb_prob={xgb_prob}, extra_trees_prob={extra_trees_prob}")
                flash('Invalid model output. Please try again or contact support.', 'error')
                return redirect(url_for('predict'))
            final_pred = 1 if (xgb_pred + extra_trees_pred) >= 1 else 0
            result = 'Diabetes Risk Detected' if final_pred == 1 else 'No Diabetes Risk'
            avg_confidence = (xgb_prob + extra_trees_prob) / 2
            analysis = generate_gemini_analysis(final_pred, avg_confidence, form_data)
            print(f"Analysis: {analysis}")
            prediction_result = {
                'result': result,
                'confidence': f'{avg_confidence:.1f}',
                'analysis': analysis
            }
            flash(f'Prediction: {result} (Confidence: {avg_confidence:.1f}%)', 'success')
            return render_template('risk_pred.html', prediction_result=prediction_result, form_data=form_data)
        except ValueError as e:
            print(f"ValueError: {str(e)}")
            flash(f'Invalid input: {str(e)}. Please ensure all fields are correctly filled.', 'error')
            return redirect(url_for('predict'))
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            flash(f'An error occurred during prediction: {str(e)}', 'error')
            return redirect(url_for('predict'))
    return render_template('risk_pred.html', prediction_result=prediction_result, form_data=form_data)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))
# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure Gemini API
genai.configure(api_key='AIzaSyC9lsET5jCJJOZmoPQ8k8TeMqeYvTvhIfk')  # Replace with your actual Gemini API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global document context
document_context = {
    'text': '',
    'metadata': {
        'title': '',
        'authors': '',
        'abstract': ''
    },
    'is_diabetes_related': False
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pdf_text(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                page_text = page.extract_text() or ''
                text += page_text
            return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        return None

def extract_metadata(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            metadata = reader.metadata
            title = metadata.get('/Title', 'Untitled Document')
            authors = metadata.get('/Author', 'Authors not specified')
            abstract = ''
            # Attempt to extract abstract from the first page
            if len(reader.pages) > 0:
                first_page_text = reader.pages[0].extract_text() or ''
                abstract_start = first_page_text.lower().find('abstract')
                if abstract_start != -1:
                    abstract_text = first_page_text[abstract_start:first_page_text.find('\n\n', abstract_start)]
                    abstract = abstract_text[:500] if len(abstract_text) > 500 else abstract_text
            return {
                'title': title,
                'authors': authors,
                'abstract': abstract or 'Abstract not available'
            }
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        return {
            'title': 'Untitled Document',
            'authors': 'Authors not specified',
            'abstract': 'Abstract not available'
        }

def analyze_diabetes_content(text):
    try:
        prompt = f"""
        Analyze the following document excerpt to determine if it is related to diabetes research. 
        If it is, provide a concise academic summary (100-150 words) of the diabetes-related content and identify 2-3 key points. 
        If not, explain why it is not relevant in a brief statement (1-2 sentences).
        Return the response in JSON format with the following structure:
        {{
            "is_diabetes_related": boolean,
            "summary": string (summary if diabetes-related, empty otherwise),
            "key_points": string (key points if diabetes-related, empty otherwise),
            "reason": string (reason if not diabetes-related, empty otherwise)
        }}
        Document Excerpt: {text[:4000]}  # Limit to avoid token issues
        """
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        result = json.loads(response_text.replace('```json\n', '').replace('\n```', ''))
        return result
    except Exception as e:
        logger.error(f"Error analyzing content with Gemini: {str(e)}")
        return {
            'is_diabetes_related': False,
            'summary': '',
            'key_points': '',
            'reason': 'Error analyzing content for diabetes relevance.'
        }

def generate_questions(text):
    try:
        prompt = f"""
        Based on the following diabetes-related document excerpt, generate relevant research questions categorized by context, gap, methodology, findings, implications, limitations, and future directions. 
        Return a JSON object with each category containing a list of 1-2 questions. 
        If the text is not diabetes-related, return an empty object.
        Document Excerpt: {text[:4000]}
        """
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        return json.loads(response_text.replace('```json\n', '').replace('\n```', ''))
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        return {}

@app.route('/diabetic_chat')
def chats():
    return render_template('chat.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            document_text = extract_pdf_text(file_path)
            if document_text is None:
                return jsonify({'error': 'Unable to extract text from PDF'}), 400
            metadata = extract_metadata(file_path)

            # Store document context temporarily
            document_context['text'] = document_text
            document_context['metadata'] = metadata

            # First analyze if it's diabetes-related
            analysis = analyze_diabetes_content(document_text)

            if not analysis['is_diabetes_related']:
                # Clear the document if not diabetes-related
                document_context['text'] = ''
                document_context['metadata'] = {
                    'title': '',
                    'authors': '',
                    'abstract': ''
                }
                document_context['is_diabetes_related'] = False
                os.remove(file_path)  # Remove the uploaded file

                return jsonify({
                    'error': analysis['reason'],
                    'cleared': True
                }), 400

            # If diabetes-related, generate questions
            questions = generate_questions(document_text)
            document_context['is_diabetes_related'] = True

            return jsonify({
                'document_text': document_text,
                'metadata': metadata,
                'questions': questions,
                'cleared': False
            })
        else:
            return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': 'Server error during file upload.'}), 500

@app.route('/analyze_diabetes', methods=['POST'])
def analyze_diabetes():
    try:
        data = request.get_json()
        document_text = data.get('document_text', '')
        if not document_text:
            return jsonify({'error': 'No document text provided'}), 400

        analysis = analyze_diabetes_content(document_text)

        if not analysis['is_diabetes_related']:
            # Automatically clear the document if it's not diabetes-related
            document_context['text'] = ''
            document_context['metadata'] = {
                'title': '',
                'authors': '',
                'abstract': ''
            }
            document_context['is_diabetes_related'] = False
            # Clear uploads folder
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))

            return jsonify({
                'is_diabetes_related': False,
                'summary': '',
                'key_points': '',
                'reason': analysis['reason'],
                'cleared': True  # Flag to indicate document was cleared
            })

        document_context['is_diabetes_related'] = analysis['is_diabetes_related']
        return jsonify({
            'is_diabetes_related': analysis['is_diabetes_related'],
            'summary': analysis['summary'],
            'key_points': analysis['key_points'],
            'reason': analysis['reason'],
            'cleared': False
        })
    except Exception as e:
        logger.error(f"Error in analyze_diabetes: {str(e)}")
        return jsonify({
            'is_diabetes_related': False,
            'summary': '',
            'key_points': '',
            'reason': 'Server error during diabetes analysis.',
            'cleared': False
        }), 500
@app.route('/chat_sync', methods=['POST'])
def chat_sync():
    try:
        data = request.get_json()
        message = data.get('message', '')
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        if document_context['is_diabetes_related']:
            prompt = f"""
            You are a diabetes based research assistant. Based on the following document context and user query, provide a concise, accurate response related to diabetes research in research and analysis tone. 
            If the query is unrelated to the document, answer using general diabetes knowledge.
            Document Context: {document_context['text'][:4000]}
            Metadata: {document_context['metadata']}
            User Query: {message}
            """
        else:
            prompt = f"""
            You are a diabetes research assistant. Answer the following user query based on general diabetes knowledge in a concise, accurate, and academic manner.
            User Query: {message}
            """
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        questions = generate_questions(document_context['text']) if document_context['is_diabetes_related'] else {}
        return jsonify({
            'response': response_text,
            'questions': questions
        })
    except Exception as e:
        logger.error(f"Error in chat_sync: {str(e)}")
        return jsonify({'error': 'Server error during chat processing.'}), 500

@app.route('/clear_document', methods=['POST'])
def clear_document():
    try:
        document_context['text'] = ''
        document_context['metadata'] = {
            'title': '',
            'authors': '',
            'abstract': ''
        }
        document_context['is_diabetes_related'] = False
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
        return jsonify({
            'success': True,
            'message': 'Document cleared successfully. Upload a new diabetes-related PDF or ask general diabetes questions.'
        })
    except Exception as e:
        logger.error(f"Error in clear_document: {str(e)}")
        return jsonify({'error': 'Server error during document clearing.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
