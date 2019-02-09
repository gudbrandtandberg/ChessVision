cd container
docker build -t chessvision-algo .
#docker run -p 5000:5000 chessvision-algo
docker run -p 8080:8080 chessvision-algo