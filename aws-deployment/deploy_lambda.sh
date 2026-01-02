#!/bin/bash
# AWS Lambda Deployment Script for California Housing Prediction API

set -e

# Configuration
REGION=${AWS_REGION:-us-east-1}
FUNCTION_NAME="california-housing-predict"
REPO_NAME="california-housing-lambda"
IMAGE_TAG="latest"

echo "🚀 Starting AWS Lambda Deployment..."

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
if [ -z "$ACCOUNT_ID" ]; then
    echo "❌ Error: Could not get AWS account ID. Make sure AWS CLI is configured."
    exit 1
fi

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

echo "📦 Account ID: ${ACCOUNT_ID}"
echo "🌍 Region: ${REGION}"
echo "📸 Image URI: ${ECR_URI}"

# Step 1: Create ECR repository if it doesn't exist
echo ""
echo "📋 Step 1: Checking ECR repository..."
if aws ecr describe-repositories --repository-names ${REPO_NAME} --region ${REGION} 2>/dev/null; then
    echo "✅ Repository ${REPO_NAME} already exists"
else
    echo "📦 Creating ECR repository..."
    aws ecr create-repository \
        --repository-name ${REPO_NAME} \
        --region ${REGION} \
        --image-scanning-configuration scanOnPush=true
    echo "✅ Repository created"
fi

# Step 2: Authenticate Docker to ECR
echo ""
echo "📋 Step 2: Authenticating Docker to ECR..."
aws ecr get-login-password --region ${REGION} | \
    docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
echo "✅ Docker authenticated"

# Step 3: Build Docker image (if not already built)
echo ""
echo "📋 Step 3: Building Docker image..."
if docker images california-housing-lambda:latest --format "{{.Repository}}:{{.Tag}}" | grep -q "california-housing-lambda:latest"; then
    echo "ℹ️  Image already exists locally, skipping build"
    echo "   To rebuild, run: docker build --platform linux/amd64 -f aws-deployment/Dockerfile.lambda -t california-housing-lambda ."
else
    echo "🔨 Building image (this may take a while)..."
    # Build from project root directory
    # Create temporary build context with required files
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    TEMP_BUILD_DIR=$(mktemp -d)
    
    # Copy required files to temp build context
    cp "$PROJECT_ROOT/aws-deployment/lambda_handler.py" "$TEMP_BUILD_DIR/"
    cp "$PROJECT_ROOT/model.pkl" "$TEMP_BUILD_DIR/"
    cp "$PROJECT_ROOT/aws-deployment/Dockerfile.lambda" "$TEMP_BUILD_DIR/Dockerfile"
    
    # Build from temp directory
    cd "$TEMP_BUILD_DIR"
    docker build --platform linux/amd64 -f Dockerfile -t california-housing-lambda .
    
    # Cleanup
    cd "$PROJECT_ROOT"
    rm -rf "$TEMP_BUILD_DIR"
    echo "✅ Image built"
    echo "✅ Image built"
fi

# Step 4: Tag and push image (only if we built locally)
echo ""
echo "📋 Step 4: Tagging and pushing image to ECR..."
if docker images california-housing-lambda:latest --format "{{.Repository}}:{{.Tag}}" 2>/dev/null | grep -q "california-housing-lambda:latest"; then
    echo "📤 Pushing locally built image..."
    docker tag california-housing-lambda:latest ${ECR_URI}
    docker push ${ECR_URI}
    echo "✅ Image pushed to ECR"
else
    echo "✅ Using existing image from ECR (built by CodeBuild)"
fi

# Step 5: Create or update Lambda function
echo ""
echo "📋 Step 5: Creating/updating Lambda function..."

# Check if function exists
if aws lambda get-function --function-name ${FUNCTION_NAME} --region ${REGION} 2>/dev/null; then
    echo "🔄 Updating existing function..."
    aws lambda update-function-code \
        --function-name ${FUNCTION_NAME} \
        --image-uri ${ECR_URI} \
        --region ${REGION} \
        --output json > /tmp/lambda-update.json
    
    # Wait for update to complete
    echo "⏳ Waiting for function update to complete..."
    aws lambda wait function-updated \
        --function-name ${FUNCTION_NAME} \
        --region ${REGION}
    
    echo "✅ Function updated"
else
    echo "🆕 Creating new function..."
    
    # Get or create execution role
    ROLE_NAME="lambda-execution-role"
    if aws iam get-role --role-name ${ROLE_NAME} 2>/dev/null; then
        ROLE_ARN=$(aws iam get-role --role-name ${ROLE_NAME} --query 'Role.Arn' --output text)
        echo "✅ Using existing role: ${ROLE_ARN}"
    else
        echo "⚠️  Role ${ROLE_NAME} not found. Creating basic execution role..."
        # Create trust policy
        cat > /tmp/trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
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
        
        # Attach basic Lambda execution policy
        aws iam attach-role-policy \
            --role-name ${ROLE_NAME} \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
        
        echo "✅ Role created: ${ROLE_ARN}"
    fi
    
    aws lambda create-function \
        --function-name ${FUNCTION_NAME} \
        --package-type Image \
        --code ImageUri=${ECR_URI} \
        --role ${ROLE_ARN} \
        --timeout 60 \
        --memory-size 3008 \
        --region ${REGION} \
        --output json > /tmp/lambda-create.json
    
    echo "✅ Function created"
fi

# Step 6: Create Function URL
echo ""
echo "📋 Step 6: Creating Function URL..."
FUNCTION_URL=$(aws lambda create-function-url-config \
    --function-name ${FUNCTION_NAME} \
    --auth-type NONE \
    --cors '{"AllowOrigins": ["*"], "AllowMethods": ["POST", "GET"], "AllowHeaders": ["content-type"]}' \
    --region ${REGION} \
    --query 'FunctionUrl' --output text 2>/dev/null || \
    aws lambda get-function-url-config \
    --function-name ${FUNCTION_NAME} \
    --region ${REGION} \
    --query 'FunctionUrl' --output text)

# Step 7: Add resource-based policy to allow public invocation
echo ""
echo "📋 Step 7: Configuring Function URL permissions..."
FUNCTION_ARN=$(aws lambda get-function --function-name ${FUNCTION_NAME} --region ${REGION} --query 'Configuration.FunctionArn' --output text)

# Add permission for public invocation (if not already exists)
aws lambda add-permission \
    --function-name ${FUNCTION_NAME} \
    --statement-id FunctionURLAllowPublicAccess \
    --action lambda:InvokeFunctionUrl \
    --principal "*" \
    --function-url-auth-type NONE \
    --region ${REGION} 2>/dev/null || echo "ℹ️  Permission already exists or was automatically created"

if [ -n "$FUNCTION_URL" ]; then
    echo "✅ Function URL: ${FUNCTION_URL}"
    echo ""
    echo "🎉 Deployment complete!"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Test the function: curl -X POST ${FUNCTION_URL}/predict -H 'Content-Type: application/json' -d @test_payload.json"
    echo "   2. Update the web page (index.html) with the Function URL: ${FUNCTION_URL}"
else
    echo "⚠️  Could not create/get Function URL"
fi

echo ""
echo "✅ Deployment script completed!"

