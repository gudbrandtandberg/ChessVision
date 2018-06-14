from flask import Flask, render_template, json, request, send_from_directory

app = Flask(__name__)

@app.route('/')
def home():
  return render_template('index.html', x=43)

@app.route("/upload", methods=['POST'])
def upload_file():
    return render_template('index.html', x="uploaded")

if __name__ == '__main__':  
  app.run(host='127.0.0.1', port=5000)
