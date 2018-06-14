import os
from flask import Flask, flash, request, redirect, url_for, make_response, current_app
from werkzeug.utils import secure_filename
from flask import send_from_directory
from datetime import timedelta
from functools import update_wrapper
import uuid 

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
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/cv_algo/', methods=['POST'])
@crossdomain(origin='*')
def predict_img():
    print("Upload invoked")
    
    file = read_file_from_formdata()

    if file is not None:
        raw_id = str(uuid.uuid4())
        filename = secure_filename(raw_id) + ".JPG"
        file_loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # The file is now saved to the upload directory
        # Next, we invoke the main program in src using the file as argument
        #res = os.system("../src/main.py -f {} -o unused".format(file_loc))

        

        #file.save(file_loc)
        

        fenfilename = os.path.join(app.config['UPLOAD_FOLDER'], filename[:-4] + "_fen.txt")
        
        with open(fenfilename, "r") as f:
            FEN = f.readline()
        
        
        ret = '{{ "FEN": "{0}", "id": "{1}" }}'.format(FEN, raw_id)

        print("Received file, sending response: {}".format(ret))
        return ret
    return '{error: true}'

@app.route('/feedback/', methods=['POST'])
@crossdomain(origin='*')
def receive_feedback():
    res = "{success: false}"

    raw_id = request.form['id']
    feedback = request.form['feedback']

    print("{0}, {1}".format(raw_id, feedback))
    res = '{success: true}'
    return res

def read_file_from_formdata():
    # check if the post request has the file part
    
    if 'file' not in request.files:
        print("No file")
        return None

    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename:
        print("No legal file selected")
        return None
    
    return file


if __name__ == '__main__':
  app.run(host='127.0.0.1', port=7777)
