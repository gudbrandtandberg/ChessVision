
image="chessvision-algo"
version="1.0"
region=eu-central-1
account=$(aws sts get-caller-identity --query Account --output text)
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${version}"

# Get the login command from ECR and execute it directly
# $(aws ecr get-login --region ${region} --no-include-email)
#aws ecr get-login-password --region eu-central-1 --profile personal | docker login --username AWS --password-stdin 580857158266.dkr.ecr.eu-central-1.amazonaws.com
# 580857158266.dkr.ecr.eu-central-1.amazonaws.com/chessvision-algo

# Build the docker image locally with the image name and then push it to ECR
docker build  -t ${image} .
docker tag ${image} ${fullname}
docker push ${fullname}