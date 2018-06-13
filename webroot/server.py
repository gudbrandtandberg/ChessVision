from flask import Flask, render_template, json, request, send_from_directory
app = Flask(__name__)

@app.route('/')
def home():
  return render_template('index.html', x=42)

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('static/js', path)

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('static/css', path)

@app.route('/img/<path:path>')
def send_img(path):
    return send_from_directory('static/img', path)

if __name__ == '__main__':
  app.run()
