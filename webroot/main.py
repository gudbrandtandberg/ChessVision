from flask import Flask, render_template
import argparse

app = Flask(__name__)

@app.route('/')
def home():
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--local", type=bool, default=True)
  args = parser.parse_args()
  local = args.local

  if local:
    endpoint = "http://localhost:7777/"
  else:
    endpoint = "http://23.97.186.93:8080/"

  return render_template('index.html', endpoint=endpoint)

if __name__ == '__main__':  
  app.run(host='127.0.0.1', port=5000)