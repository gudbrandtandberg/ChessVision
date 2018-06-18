import os
from flask import Flask, flash, request, redirect, url_for, make_response, current_app
from werkzeug.utils import secure_filename
from flask import send_from_directory
from datetime import timedelta
from functools import update_wrapper
import uuid
import chessvision
import cv2
import numpy as np

from board_extractor import load_extractor
from board_classifier import load_classifier

from util import BoardExtractionError

app = Flask(__name__)

def crossdomain(origin=None, methods=None, headers=None, max_age=21600,
                attach_to_all=True, automatic_options=True):
    """Decorator function that allows crossdomain requests.
      Courtesy of
      https://blog.skyred.fi/articles/better-crossdomain-snippet-for-flask.html
    """
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, str):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, str):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        """ Determines which methods are allowed
        """
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        """The decorator function
        """
        def wrapped_function(*args, **kwargs):
            """Caries out the actual cross domain code
            """
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator


UPLOAD_FOLDER = './user_uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TMP_FOLDER'] = "./tmp/"
app.secret_key = 'super secret key'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/cv_algo/', methods=['POST'])
@crossdomain(origin='*')
def predict_img():
    print("CV-Algo invoked")
    
    file = read_file_from_formdata()

    if file is not None:
        raw_id = str(uuid.uuid4())
        filename = secure_filename(raw_id) + ".JPG"
        tmp_loc = os.path.join(app.config['TMP_FOLDER'], filename)
        
        #print("Will classfiy file stored at {}".format(tmp_loc))
        file.save(tmp_loc)
        tmp_path = os.path.abspath(tmp_loc)

        # check if input image is flipped (from black's perspective)
        flip = False
        if "reversed" in request.form:
            flip = True
        print(flip)
        try:
            
            board_img, predictions, FEN, _ = chessvision.classify_raw(tmp_path, board_model, sq_model)
            #move file to success raw folder
            os.rename(tmp_loc, os.path.join("./user_uploads/raw_success/", filename))
            cv2.imwrite("./user_uploads/unlabelled/boards/x_" + filename, board_img)
            np.save("./user_uploads/unlabelled/predictions/"+raw_id+".npy", predictions)
        except BoardExtractionError as e:
            #move file to success raw folder
            os.rename(tmp_loc, os.path.join("./user_uploads/raw_fail/", filename))
            return e.json_string()

        ret = '{{ "FEN": "{0}", "id": "{1}", "error": "false" }}'.format(FEN, raw_id)

        return ret
    
    return '{"error": "true", "errorMsg": "Fuck!"}'


@app.route('/feedback/', methods=['POST'])
@crossdomain(origin='*')
def receive_feedback():
    res = '{"success": "false"}'

    raw_id = request.form['id']
    feedback = request.form['feedback']
    
    board_filename = "x_" + raw_id + ".JPG"
    pred_filename = raw_id + ".npy"

    board_src = os.path.join("./user_uploads/unlabelled/boards/", board_filename)
    pred_src = os.path.join("./user_uploads/unlabelled/predictions/", pred_filename)
    
    try:
        if feedback == "correct":
            board_dst = os.path.join("./user_uploads/labelled/boards/", board_filename)
            pred_dst = os.path.join("./user_uploads/labelled/predictions/", pred_filename)        
        elif feedback == "incorrect":
            board_dst = os.path.join("./user_uploads/failboards/", board_filename)
            pred_dst = None # don't save incorrect predictions
        else:
            return '{"success": "false"}'

        ## Move unlabelled board
        print("Moving {} to {}".format(board_src, board_dst))
        os.rename(board_src, board_dst)
        
        ## Move or delete predictions file
        if pred_dst is None:
            print("Deleting {}".format(pred_src))
            os.remove(pred_src)
        else:
            print("Moving {} to {}".format(pred_src, pred_dst))
            os.rename(pred_src, pred_dst)
        
        res = '{"success": "true"}'

    except OSError as e:
        return res

    return res


def read_file_from_formdata():
    # check if the post request has the file part
    
    if 'file' not in request.files:
        print("No file")
        return None

    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        print("No legal file selected")
        return None
    
    return file


def load_models():
    global sq_model, board_model

    sq_model = load_classifier()
    board_model = load_extractor()

    return board_model, sq_model

if __name__ == '__main__':

    board_model, sq_model = load_models()
    app.run(host='127.0.0.1', port=7777)
