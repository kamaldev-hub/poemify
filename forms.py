from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField, IntegerField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo, Length, NumberRange
from flask_wtf.file import FileField, FileAllowed, FileRequired


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Sign In')


class PoemGenerationForm(FlaskForm):
    prompt = TextAreaField('Prompt', validators=[DataRequired()])
    versions = IntegerField('Number of Versions', validators=[NumberRange(min=1, max=5)], default=1)
    style = SelectField('Style', validators=[DataRequired()])
    submit = SubmitField('Generate Poem')


class UploadForm(FlaskForm):
    file = FileField('File', validators=[
        FileRequired(),
        FileAllowed(['txt', 'json', 'pdf'], 'Only TXT, JSON, or PDF files are allowed!')
    ])
    submit = SubmitField('Upload')
