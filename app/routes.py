from app import app
from flask import render_template, redirect, url_for, request
from app.forms import SentenceForm
from models.inference import sent_inference

@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"


@app.route('/inference', methods=['GET', 'POST'])
def inference():
    form = SentenceForm()
    output = ""

    base_color = '0x00000'
    target_color = '0xff0000'

    if form.validate_on_submit():
        # Perform inference
        result = sent_inference(form.source_string.data, encoder_no=form.source_option.data)

        colors = []
        for _, conf in result:
            colors.append(hex(round(conf*int(target_color, 16))))
        #output = [(word, color) for word, conf in result for color in colors]
        output = list(zip(result, colors))

        print(output)
    return render_template('sentence_input.html', form=form, output=output)
