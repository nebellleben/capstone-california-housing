#!/bin/bash
# Setup AWS CodeBuild directly (without CloudFormation)

set -e

REGION=${AWS_REGION:-us-east-1}
PROJECT_NAME="california-housing-lambda-build"
ECR_REPO_NAME="california-housing-lambda"
S3_BUCKET_NAME="${PROJECT_NAME}-source-$(date +%s)"

echo "🚀 Setting up AWS CodeBuild for Lambda deployment..."

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
if [ -z "$ACCOUNT_ID" ]; then
    echo "❌ Error: Could not get AWS account ID"
    exit 1
fi

# Step 1: Create ECR repository
echo ""
echo "📦 Step 1: Creating ECR repository..."
if aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${REGION} 2>/dev/null; then
    echo "✅ Repository ${ECR_REPO_NAME} already exists"
else
    aws ecr create-repository \
        --repository-name ${ECR_REPO_NAME} \
        --region ${REGION} \
        --image-scanning-configuration scanOnPush=true
    echo "✅ Repository created"
fi

# Step 2: Create S3 bucket
echo ""
echo "📦 Step 2: Creating S3 bucket for source code..."
if [ "$REGION" == "us-east-1" ]; then
    aws s3 mb "s3://${S3_BUCKET_NAME}" 2>/dev/null || echo "Bucket may already exist"
else
    aws s3 mb "s3://${S3_BUCKET_NAME}" --region "${REGION}" 2>/dev/null || echo "Bucket may already exist"
fi
echo "✅ Bucket: ${S3_BUCKET_NAME}"

# Step 3: Create source package
echo ""
echo "📦 Step 3: Creating source code package..."
ZIP_FILE="/tmp/california-housing-source.zip"
rm -f ${ZIP_FILE}

# Create zip with required files
# We need to create a flat structure in the zip for CodeBuild
cd "$(dirname "$0")/.." || exit 1
TEMP_DIR=$(mktemp -d)
# Copy files to temp directory in flat structure
cp aws-deployment/Dockerfile.lambda ${TEMP_DIR}/
cp aws-deployment/lambda_handler.py ${TEMP_DIR}/
cp aws-deployment/buildspec.yml ${TEMP_DIR}/
cp pyproject.toml ${TEMP_DIR}/
cp uv.lock ${TEMP_DIR}/
cp model.pkl ${TEMP_DIR}/
# Create zip from temp directory
cd ${TEMP_DIR}
zip -r ${ZIP_FILE} .
cd - > /dev/null
rm -rf ${TEMP_DIR}
cd - > /dev/null

echo "✅ Source package created"

# Step 4: Upload to S3
echo ""
echo "📤 Step 4: Uploading source to S3..."
S3_KEY="source/california-housing-source.zip"
aws s3 cp ${ZIP_FILE} "s3://${S3_BUCKET_NAME}/${S3_KEY}"
echo "✅ Source uploaded"

# Step 5: Create IAM role for CodeBuild
echo ""
echo "🔐 Step 5: Creating IAM role..."
ROLE_NAME="${PROJECT_NAME}-role"

# Check if role exists
if aws iam get-role --role-name ${ROLE_NAME} 2>/dev/null; then
    echo "✅ Role ${ROLE_NAME} already exists"
    ROLE_ARN=$(aws iam get-role --role-name ${ROLE_NAME} --query 'Role.Arn' --output text)
else
    # Create trust policy
    cat > /tmp/trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "codebuild.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    ROLE_ARN=$(aws iam create-role \
        --role-name ${ROLE_NAME} \
        --assume-role-policy-document file:///tmp/trust-policy.json \
        --query 'Role.Arn' --output text)

    # Attach policies
    cat > /tmp/codebuild-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/aws/codebuild/${PROJECT_NAME}*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:GetAuthorizationToken",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::${S3_BUCKET_NAME}/*"
    }
  ]
}
EOF

    aws iam put-role-policy \
        --role-name ${ROLE_NAME} \
        --policy-name CodeBuildPolicy \
        --policy-document file:///tmp/codebuild-policy.json

    echo "✅ Role created: ${ROLE_ARN}"
fi

# Step 6: Create CodeBuild project
echo ""
echo "🔨 Step 6: Creating CodeBuild project..."

# Check if project exists
if aws codebuild batch-get-projects --names ${PROJECT_NAME} --region ${REGION} 2>&1 | grep -q "not found\|does not exist"; then
    CREATE_PROJECT=true
else
    CREATE_PROJECT=false
    echo "ℹ️  Project exists, updating..."
    aws codebuild update-project \
        --name ${PROJECT_NAME} \
        --source type=S3,location=${S3_BUCKET_NAME}/${S3_KEY} \
        --region ${REGION} > /dev/null 2>&1 && echo "✅ Project updated" || CREATE_PROJECT=true
fi

if [ "$CREATE_PROJECT" = true ]; then
    # Create project configuration
    cat > /tmp/codebuild-project.json <<EOF
{
  "name": "${PROJECT_NAME}",
  "description": "Build Docker image for California Housing Lambda function",
  "source": {
    "type": "S3",
    "location": "${S3_BUCKET_NAME}/${S3_KEY}",
    "buildspec": "buildspec.yml"
  },
  "artifacts": {
    "type": "NO_ARTIFACTS"
  },
  "environment": {
    "type": "LINUX_CONTAINER",
    "image": "aws/codebuild/standard:7.0",
    "computeType": "BUILD_GENERAL1_LARGE",
    "privilegedMode": true,
    "environmentVariables": [
      {
        "name": "ECR_REPO_NAME",
        "value": "${ECR_REPO_NAME}"
      },
      {
        "name": "IMAGE_TAG",
        "value": "latest"
      },
      {
        "name": "AWS_DEFAULT_REGION",
        "value": "${REGION}"
      }
    ]
  },
  "serviceRole": "${ROLE_ARN}",
  "timeoutInMinutes": 60,
  "logsConfig": {
    "cloudWatchLogs": {
      "status": "ENABLED",
      "groupName": "/aws/codebuild/${PROJECT_NAME}"
    }
  }
}
EOF

    aws codebuild create-project \
        --cli-input-json file:///tmp/codebuild-project.json \
        --region ${REGION} > /dev/null
    echo "✅ Project created"
fi

# Step 7: Start build
echo ""
echo "🔨 Step 7: Starting CodeBuild..."
BUILD_ID=$(aws codebuild start-build \
    --project-name ${PROJECT_NAME} \
    --region ${REGION} \
    --query 'build.id' --output text)

echo "✅ Build started: ${BUILD_ID}"
echo ""
echo "📊 Monitor build:"
echo "   https://${REGION}.console.aws.amazon.com/codesuite/codebuild/projects/${PROJECT_NAME}/build/${BUILD_ID}"
echo ""
echo "⏳ Build will take ~15-20 minutes. Checking status..."

# Wait and show status
for i in {1..40}; do
    STATUS=$(aws codebuild batch-get-builds \
        --ids ${BUILD_ID} \
        --region ${REGION} \
        --query 'builds[0].buildStatus' --output text 2>/dev/null || echo "IN_PROGRESS")
    
    if [ "$STATUS" == "SUCCEEDED" ]; then
        echo ""
        echo "✅ Build completed successfully!"
        break
    elif [ "$STATUS" == "FAILED" ] || [ "$STATUS" == "FAULT" ] || [ "$STATUS" == "TIMED_OUT" ] || [ "$STATUS" == "STOPPED" ]; then
        echo ""
        echo "❌ Build failed with status: ${STATUS}"
        echo "   Check logs: https://${REGION}.console.aws.amazon.com/cloudwatch/home?region=${REGION}#logsV2:log-groups/log-group//aws/codebuild/${PROJECT_NAME}"
        exit 1
    else
        echo -n "."
        sleep 30
    fi
done

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}:latest"
echo ""
echo "🎉 CodeBuild setup complete!"
echo ""
echo "📝 ECR Image: ${ECR_URI}"
echo "📝 Next: Run ./deploy_lambda.sh to deploy Lambda function"

