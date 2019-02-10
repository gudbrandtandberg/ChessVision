import boto3
import json
import requests

local = True

with open("sagemaker/container/local_test/test_img.txt", "r") as f:
    image = f.read()

payload = json.dumps({
    "image": image,
    "flip": "false"
})

if local:

    r1 = requests.get("http://127.0.0.1:8080/ping", data="")
    r2 = requests.post("http://127.0.0.1:8080/invocations", data=payload, headers={"Content-Type": "application/json"})

    print(r1)
    print(r2)

else:
    client = boto3.client('sagemaker-runtime')
    custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"  
    endpoint_name = "chessvision-endpoint"                      
    content_type = "application/json"                           
    accept = "text/plain"                                   

    response = client.invoke_endpoint(
        EndpointName=endpoint_name, 
        CustomAttributes=custom_attributes, 
        ContentType=content_type,
        Accept=accept,
        Body=payload
        )

    print(response)