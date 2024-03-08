# app.py
from flask import Flask, render_template, request, url_for
import cv2 as cv
import numpy as np
import cv_utils
from yolo_classifier import color_classifier
from yolo_classifier import brand_classifier


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload():
    if request.method == 'POST': 
        # Pega a lista de arquivos vindos do front-end
        uploaded_pictures = request.files.getlist("file") 
 
        # Verifica se a lista foi enviada vazia
        if len(uploaded_pictures) == 0 or uploaded_pictures[0].filename == '':
            return render_template('index.html')   
        
        pipeline_results = []
        for raw_picture in uploaded_pictures:
            image = cv.imdecode(np.frombuffer(raw_picture.read(), np.uint8), cv.IMREAD_COLOR)
            #height, width, _ = image.shape
            # filename = raw_picture.filename

            # Codificação da imagem para .jpg e base64 para o front-end
            # image_64 = cv_utils.cv_to_base64(image)
            #images_64.append(image_64)
            
            # height, width, _ = image.shape
            # resized_image = cv.resize(image, (640, 640))

            # Classificação por cores
            img_color_classified = color_classifier(image)
            img_color_classified_64 = cv_utils.cv_to_base64(img_color_classified)

            # Classificação por marcas
            img_brand_clasified = brand_classifier(image)
            img_brand_clasified_64 = cv_utils.cv_to_base64(img_brand_clasified)
                
            pipeline_results.append(
                {
                'img_brand_clasified_64': img_brand_clasified_64,
                'img_color_classified_64': img_color_classified_64,
            })

    return render_template(
        'resultado.html',
        pipeline_results=pipeline_results
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
