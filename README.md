# Poemify: AI-Powered Poetry Generator

**Poemify** is an AI-driven web application that generates poetry based on user-uploaded documents and prompts. It analyzes the style of uploaded poems and produces new poems in a similar tone and style.

## ✨ Features

- **User Registration & Authentication**: Secure user accounts.
- **Document Upload & Analysis**: Analyze documents containing poetry.
- **AI-Powered Poem Style Detection**: Detect the style of uploaded poems.
- **Custom Poem Generation**: Create new poems based on user prompts and detected styles.
- **Poem Management Dashboard**: Manage and view generated poems.

## 🚀 Technologies Used

- **Backend**: Python 3.8+, Flask, SQLAlchemy, Flask-Login
- **AI Integration**: Groq API for text generation, FAISS for similarity search
- **Document Processing**: PyPDF2 for PDF text extraction

## 🛠️ Setup Instructions

Follow these steps to set up the project on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/kamaldev-hub/poemify.git
cd poemify
```
### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a .env file in the project root and add the following:
```bash
SECRET_KEY=your_secret_key
GROQ_API_KEY=your_groq_api_key
```
### 5. Initialize the Database
```bash
flask db init
flask db migrate
flask db upgrade
```
### 6. Run the Server
```bash
Run the app.py file
```
## 📖 Usage

1. **Register** a new account or **log in** to your existing one.
2. **Upload** a document (PDF) containing poems for analysis.
3. **Generate** new poems by providing prompts and selecting a style based on the uploaded documents.
4. **Manage** and view all generated poems in your personal dashboard.