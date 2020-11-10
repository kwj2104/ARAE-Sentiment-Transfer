from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField
from wtforms.validators import DataRequired, Optional, ValidationError

def validate_staff(form, field):
    if field.data == 0:
        raise ValidationError("Sorry, you havn't chosen a option")

class SentenceForm(FlaskForm):
    options = [(0, "--"), (2, 'Negative'), (1,'Positive')]
    source_string = StringField('Insert Review Here', validators=[DataRequired()])
    source_option = SelectField('Sentiment', choices=options, coerce=int, validators=[validate_staff])
    submit = SubmitField('Convert')
