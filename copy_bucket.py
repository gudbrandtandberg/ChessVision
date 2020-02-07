import os
import boto3

client = boto3.client("s3")

avoid = [
    "requirements.txt",
    "web.2.7.config",
    "ptvs_virtualenv_proxy.py",
    "runtime.txt",
    "README.md",
    "copy_bucket.py",
    "main.py"
]
rootDir = 'webroot'

for dirName, subdirList, fileList in os.walk(rootDir):
    if "." in dirName:
        continue
    for fname in fileList:
        contentType = "text/plain"
        if fname[0] == "." or fname in avoid:
            continue
        if fname.endswith(".html"):
            contentType = "text/html"
        if fname.lower().endswith(".jpg"):
            contentType = "image/jpeg"
        if fname.endswith("js"):
            contentType = "text/javascript"
        if fname.endswith(".css"):
            contentType = "text/css"
        if fname.endswith(".png"):
            contentType = "image/png"
        if fname.endswith(".ico"):
            contentType = "image/x-icon"
        
        key = "{}/{}".format(dirName, fname).replace("webroot/", "")
        print(key)
        f = os.path.join(dirName, fname)
        with open (f, "rb") as fHandle:
            client.put_object(Bucket="chessvision-web", Key=key, Body=fHandle, ContentType=contentType)        