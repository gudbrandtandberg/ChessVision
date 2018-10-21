from flask import Flask, render_template
import argparse

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--local", action="store_true")
args = parser.parse_args()

@app.route('/')
def home():
  
  endpoint = "http://localhost:7777/" if args.local else "http://23.97.186.93:8080/"

  return render_template('index.html', endpoint=endpoint)

if __name__ == '__main__':  

  print(args.local)
  
  app.run(host='127.0.0.1', port=5000)