
image="chessvision-algo"

chmod +x $CVROOT/container/serve

account=$(aws sts get-caller-identity --query Account --output text)
region=$(aws configure get region)
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image}"# > /dev/null 2>&1


# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
docker build  -t ${image} .
docker tag ${image} ${fullname}
docker push ${fullname}