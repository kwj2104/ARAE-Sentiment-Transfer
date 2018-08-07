from app import app
from flask import render_template, redirect, url_for
from app.forms import SentenceForm
from models.inference import sent_inference

@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"


@app.route('/inference', methods=['GET', 'POST'])
def inference():
    form = SentenceForm()
    output = "Test Default"
    if form.validate_on_submit():
        # Perform inference
        output = sent_inference(form.source_string.data, encoder_no=1)

        #return redirect(url_for('inference'))
    return render_template('sentence_input.html', form=form, output=output)
