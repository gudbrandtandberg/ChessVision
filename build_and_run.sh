docker build -t chessvision-algo .
docker run -d -p 8080:8080  --env-file .env chessvision-algo