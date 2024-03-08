import cv2 as cv
import base64
import numpy as np
import os


# Codificação Base64 para envio ao front-end
def cv_to_base64(image):
    _, buffer = cv.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def directory_cleaner(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            path_to_file = os.path.join(directory_path, file)
            os.remove(path_to_file)
        print(f"Arquivos no diretório {directory_path} foram removidos com sucesso.")
    except Exception as e:
        print(f"Ocorreu um erro ao tentar limpar o diretório {directory_path}: {e}")