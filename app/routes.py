from app import app
from flask import render_template, redirect, url_for, request
from app.forms import SentenceForm
from models.inference import sent_inference

@app.route('/')
@app.route('/index')
def index():
    return "Hello, World! TEST"


@app.route('/inference', methods=['GET', 'POST'])
def inference():
    form = SentenceForm()
    output = ""

    base_color = '0x00000'
    target_color = '0xff0000'

    if form.validate_on_submit():
        # Perform inference
        result_25 = sent_inference(form.source_string.data, encoder_no=form.source_option.data, state_dict="models/trained_models/autoencoder_model_25.pt")
        result_50 = sent_inference(form.source_string.data, encoder_no=form.source_option.data, state_dict="models/trained_models/autoencoder_model_50.pt")
        result_l10 = sent_inference(form.source_string.data, encoder_no=form.source_option.data, state_dict="models/trained_models/autoencoder_model_lambda10_50.pt")

        colors = []
        for _, conf in result_25:
            colors.append(hex(round(conf*int(target_color, 16))))
        output_25 = list(zip(result_25, colors))

        for _, conf in result_50:
            colors.append(hex(round(conf*int(target_color, 16))))
        output_50 = list(zip(result_25, colors))

        for _, conf in result_l10:
            colors.append(hex(round(conf*int(target_color, 16))))
        output_l10 = list(zip(result_25, colors))

        print(output_25)
        print(output_50)
        print(output_l10)

    return render_template('sentence_input.html', form=form, output1=output_25, output2=output_50, output3=output_l10)
