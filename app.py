from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
import os
import json
import PyPDF2
from models import db, User, Poem, UploadedDocument
from forms import RegistrationForm, LoginForm, PoemGenerationForm, UploadForm
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = Flask(__name__)
app.config.from_object('config.Config')

# Set up upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'error')
    return render_template('login.html', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    poems = current_user.poems.order_by(Poem.created_at.desc()).all()
    return render_template('dashboard.html', poems=poems)


@app.route('/upload_document', methods=['GET', 'POST'])
@login_required
def upload_document():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.file.data
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Extract text from the uploaded file
            content = extract_text_from_file(file_path)

            # Detect style using LLM
            detected_style, structure_info = detect_style(content, filename)

            # Create chunks from the content
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(content)

            # Create vector store
            vectorstore = FAISS.from_texts(chunks, embeddings)

            # Save the vectorstore
            vectorstore_path = os.path.join(app.config['UPLOAD_FOLDER'], f"vectorstore_{current_user.id}_{filename}")
            vectorstore.save_local(vectorstore_path)

            # Save document info to database
            new_doc = UploadedDocument(user_id=current_user.id, filename=filename, vectorstore_path=vectorstore_path,
                                       detected_style=detected_style, structure_info=json.dumps(structure_info))
            db.session.add(new_doc)
            db.session.commit()

            os.remove(file_path)  # Remove the original file after processing
            flash(f'Document uploaded and processed successfully. Detected style: {detected_style}', 'success')
            return redirect(url_for('dashboard'))
    return render_template('upload_document.html', form=form)


@app.route('/generate_poem', methods=['GET', 'POST'])
@login_required
def generate_poem_route():
    form = PoemGenerationForm()
    user_docs = UploadedDocument.query.filter_by(user_id=current_user.id).all()
    form.style.choices = [(str(doc.id), f"{doc.filename} ({doc.detected_style})") for doc in user_docs]

    if form.validate_on_submit():
        poems = []
        selected_doc = UploadedDocument.query.get(int(form.style.data))
        vectorstore = FAISS.load_local(selected_doc.vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        structure_info = json.loads(selected_doc.structure_info)

        for _ in range(form.versions.data):
            # Generate poem using LLM and RAG
            poem_text = generate_poem(form.prompt.data, selected_doc.detected_style, vectorstore, structure_info)

            # Post-process the generated poem
            poem_text = post_process_poem(poem_text, structure_info)

            new_poem = Poem(user_id=current_user.id, prompt=form.prompt.data, style=selected_doc.detected_style,
                            generated_poem=poem_text)
            db.session.add(new_poem)
            poems.append(new_poem)
        db.session.commit()
        return render_template('generated_poems.html', poems=poems)
    return render_template('generate_poem.html', form=form)


@app.route('/get_poem/<int:poem_id>')
@login_required
def get_poem(poem_id):
    poem = Poem.query.get_or_404(poem_id)
    if poem.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify({
        "id": poem.id,
        "prompt": poem.prompt,
        "style": poem.style,
        "generated_poem": poem.generated_poem,
        "created_at": poem.created_at.strftime('%Y-%m-%d %H:%M')
    })


@app.route('/delete_poem/<int:poem_id>', methods=['POST'])
@login_required
def delete_poem(poem_id):
    poem = Poem.query.get_or_404(poem_id)
    if poem.user_id != current_user.id:
        flash('You do not have permission to delete this poem.', 'error')
        return redirect(url_for('dashboard'))

    db.session.delete(poem)
    db.session.commit()
    flash('Poem deleted successfully.', 'success')
    return redirect(url_for('dashboard'))


def detect_style(content, filename):
    client = Groq(api_key=app.config['GROQ_API_KEY'])

    content_sample = extract_poem_sample(content)
    structure_info = analyze_poem_structure(content)

    prompt = f"""Analyze the following poem sample, filename, and structural information to determine the specific poetic style:

Filename: {filename}

Sample content:
{content_sample}

Structural Information:
Average line count: {structure_info['avg_line_count']}
Most common line count: {structure_info['most_common_line_count']}
Most common rhyme scheme: {structure_info['most_common_rhyme_scheme']}
Average syllables per line: {structure_info['avg_syllables_per_line']}

Provide a concise name for the poetic style, considering the content, filename, and structural information. 
Focus on identifying specific forms or styles associated with particular poets or movements.
Your response should be no more than 3 words."""

    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an expert in poetic styles with a deep knowledge of literary history and forms."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=20,
        top_p=1,
        stream=False,
        stop=None
    )
    detected_style = completion.choices[0].message.content.strip()

    # Ensure the style is no more than 3 words
    detected_style = ' '.join(detected_style.split()[:3])

    return detected_style, structure_info


def extract_poem_sample(content):
    lines = content.split('\n')
    poem_lines = [line.strip() for line in lines if
                  line.strip() and not line.strip().startswith(('Contents', 'Chapter', 'Page'))]
    return '\n'.join(poem_lines[:20])


def analyze_poem_structure(content):
    lines = content.split('\n')
    poems = []
    current_poem = []
    for line in lines:
        if line.strip():
            current_poem.append(line.strip())
        elif current_poem:
            poems.append(current_poem)
            current_poem = []
    if current_poem:
        poems.append(current_poem)

    line_counts = [len(poem) for poem in poems]
    avg_line_count = sum(line_counts) / len(line_counts) if line_counts else 0
    most_common_line_count = max(set(line_counts), key=line_counts.count)

    rhyme_schemes = [analyze_rhyme_scheme(poem) for poem in poems]
    most_common_rhyme_scheme = max(set(rhyme_schemes), key=rhyme_schemes.count)

    # Analyze meter (simplified)
    sample_lines = [line for poem in poems for line in poem[:5]]  # Take first 5 lines of each poem
    avg_syllables = sum(count_syllables(line) for line in sample_lines) / len(sample_lines)

    return {
        "avg_line_count": avg_line_count,
        "most_common_line_count": most_common_line_count,
        "most_common_rhyme_scheme": most_common_rhyme_scheme,
        "avg_syllables_per_line": avg_syllables
    }


def count_syllables(line):
    # This is a very simplified syllable counter. For a more accurate count,
    # you might want to use a library like nltk or create a more sophisticated algorithm.
    return len(line.split())


def analyze_rhyme_scheme(poem):
    last_words = [line.split()[-1].lower() for line in poem if line.split()]
    rhyme_scheme = ""
    rhyme_dict = {}
    current_letter = 'A'
    for word in last_words:
        if word not in rhyme_dict:
            rhyme_dict[word] = current_letter
            current_letter = chr(ord(current_letter) + 1)
        rhyme_scheme += rhyme_dict[word]
    return rhyme_scheme


def generate_poem(prompt, style, vectorstore, structure_info):
    # Retrieve relevant context using the vectorstore
    relevant_chunks = vectorstore.similarity_search(prompt, k=3)
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])

    client = Groq(api_key=app.config['GROQ_API_KEY'])
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are a skilled poet specializing in the following style:
{style}

Use the following context from similar poems to inform your writing:
{context}

Adhere to these structural requirements:
- Number of lines: {structure_info['most_common_line_count']}
- Rhyme scheme: {structure_info['most_common_rhyme_scheme']}
- Average syllables per line: {structure_info['avg_syllables_per_line']}

Write a new poem based on the given prompt, emulating the style and structure of the provided context."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=300,
        top_p=0.9,
        stream=False,
        stop=None
    )
    return completion.choices[0].message.content.strip()


def post_process_poem(poem, structure_info):
    lines = poem.split('\n')
    target_lines = int(structure_info['most_common_line_count'])
    lines = lines[:target_lines]
    while len(lines) < target_lines:
        lines.append("[Line missing]")  # Placeholder for missing lines

    return '\n'.join(lines)


def extract_text_from_file(file_path):
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            return f.read()
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return '\n'.join([poem.get('content', '') for poem in data])
    elif file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    else:
        raise ValueError("Unsupported file type")


def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages])


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)