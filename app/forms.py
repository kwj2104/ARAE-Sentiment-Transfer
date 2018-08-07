from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class SentenceForm(FlaskForm):
    source_string = StringField('Insert Review Here', validators=[DataRequired()])
    submit = SubmitField('Convert')
