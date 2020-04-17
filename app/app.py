import os, sys
import sys
sys.path.append("..")
sys.path.append(".")

from flask import Flask, escape, request,  Response, g, make_response
from flask.templating import render_template
from werkzeug.utils import secure_filename
# from . import neural_style_transfer
import inference
app = Flask(__name__)
app.debug = True
# app.config['SECRET_KEY'] = '어려운 암호 키'


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/nst_get')
def nst_get():

    return render_template('nst_get.html')

@app.route('/nst_post', methods=['GET','POST'])
def nst_post():
    if request.method == 'POST':
        # Reference Image
        # refer_img = request.form['refer_img']
        # refer_img_path = 'static/images/' + str(refer_img)

        # User Image (target image)
        user_img = request.files['user_img']
        # user_img_path  '/root/optimization/ocrSecurity/app/static/images/' + str(user_img.filename)
        user_img_path = 'images/' + str(user_img.filename)
        print("user_img_path",user_img)
        user_img.save('/root/optimization/ocrSecurity/app/static/images/' + str(user_img.filename))

        # L7 OCR
        output_img = ocr_server.infer('/root/optimization/ocrSecurity/app/static/images/',str(user_img.filename))
        # transfer_img_path = './static/images/' + str(transfer_img.split('/')[-1])

    return render_template('nst_post.html',
                           refer_img=user_img_path, user_img='images/output/'+output_img)


if __name__ == "__main__":
    ocr_server = inference.inference_obj()
    app.run(host='0.0.0.0', port=8237)
