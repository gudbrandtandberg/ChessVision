import requests

with open("local_test/test.json", "r") as f:
    payload = f.read()

r1 = requests.get("http://127.0.0.1:8080/ping", data="")
r2 = requests.post("http://127.0.0.1:8080/invocations", data=payload, headers={"Content-Type": "application/json"})

print(r1.text)
print(r2.text)
