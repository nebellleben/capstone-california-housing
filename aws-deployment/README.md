# AWS Deployment

This directory contains all files needed to deploy the California Housing Prediction API to AWS Lambda, including CodeBuild, ECR, Lambda, and S3 website hosting.

## Quick Start

From the **project root directory**:

```bash
# 1. Setup CodeBuild (one-time setup)
./aws-deployment/setup_codebuild_direct.sh

# 2. Deploy Lambda function
./aws-deployment/deploy_lambda.sh

# 3. Deploy website
./aws-deployment/deploy_website.sh
```

## Files Overview

### Essential Deployment Scripts
- **`deploy_lambda.sh`** - Deploy Lambda function from ECR image
- **`deploy_website.sh`** - Deploy test website to S3
- **`setup_codebuild_direct.sh`** - Setup CodeBuild project (one-time)

### Configuration Files
- **`Dockerfile.lambda`** - Dockerfile for Lambda container image
- **`buildspec.yml`** - CodeBuild build specification
- **`lambda_handler.py`** - Lambda-compatible FastAPI handler (uses Mangum)
- **`index.html`** - Test website for the prediction API

### Utility Scripts
- **`fix_function_url_complete.sh`** - Fix Function URL 403 errors (delete and recreate)

## Deployment Process

### Prerequisites
- AWS CLI configured with appropriate credentials
- AWS account with permissions for: CodeBuild, ECR, Lambda, S3, IAM
- Files in project root: `model.pkl`, `pyproject.toml`, `uv.lock`

### Step 1: Setup CodeBuild (One-Time)

This creates the CodeBuild project that will build your Docker image:

```bash
./aws-deployment/setup_codebuild_direct.sh
```

This script will:
- Create ECR repository
- Create S3 bucket for source code
- Create CodeBuild project
- Start the build process

### Step 2: Deploy Lambda Function

After CodeBuild successfully builds the image, deploy it to Lambda:

```bash
./aws-deployment/deploy_lambda.sh
```

This script will:
- Use the Docker image from ECR (built by CodeBuild)
- Create/update Lambda function
- Configure Function URL with public access
- Add necessary permissions

### Step 3: Deploy Website

Deploy the test website to S3:

```bash
./aws-deployment/deploy_website.sh
```

Or specify a bucket name:
```bash
./aws-deployment/deploy_website.sh my-bucket-name
```

## Lambda Configuration

The Lambda function is configured with:
- **Memory**: 3008 MB (required for TensorFlow)
- **Timeout**: 60 seconds
- **Architecture**: x86_64
- **Package Type**: Container Image
- **Handler**: `lambda_handler.handler`

## Function URL

After deployment, the Function URL is automatically created with:
- **Auth Type**: NONE (public access)
- **CORS**: Enabled for all origins
- **Resource Policy**: Public access permission added

### Fixing Function URL Issues

If you encounter 403 Forbidden errors:

```bash
./aws-deployment/fix_function_url_complete.sh
```

This script will delete and recreate the Function URL with proper permissions.

## Testing

### Test the API directly:

```bash
# Get Function URL
FUNCTION_URL=$(aws lambda get-function-url-config \
    --function-name california-housing-predict \
    --region us-east-1 \
    --query 'FunctionUrl' --output text)

# Health check
curl -X GET "${FUNCTION_URL}/"

# Make prediction
curl -X POST "${FUNCTION_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }'
```

### Test via Website

After deploying the website, visit:
```
http://california-housing-predictor.s3-website-us-east-1.amazonaws.com
```

Enter your Function URL in the configuration section and test predictions.

## File Dependencies

The deployment process requires these files from the **project root directory**:
- `model.pkl` - Trained machine learning model
- `pyproject.toml` - Python project configuration
- `uv.lock` - Dependency lock file

These are automatically included when:
- Building Docker images (via `Dockerfile.lambda`)
- Creating CodeBuild source packages (via `setup_codebuild_direct.sh`)

## Important Notes

⚠️ **All scripts must be run from the project root directory**, not from within `aws-deployment/`.

The scripts automatically handle paths to reference files in the parent directory.

### Build Context

- **CodeBuild**: Files are extracted to a flat structure after zip extraction
- **Local Builds**: `deploy_lambda.sh` creates a temporary build context with required files

## Troubleshooting

### Function URL Returns 403 Forbidden

Run the fix script:
```bash
./aws-deployment/fix_function_url_complete.sh
```

### Lambda Function Errors

1. Check CloudWatch Logs:
   ```bash
   aws logs tail /aws/lambda/california-housing-predict --follow --region us-east-1
   ```

2. Verify model.pkl is in the image
3. Check memory and timeout settings (should be 3008 MB, 60 seconds)

### CodeBuild Fails

1. Check build logs in AWS Console or via:
   ```bash
   aws codebuild batch-get-builds --ids <build-id> --region us-east-1
   ```

2. Verify all required files are in the source package
3. Check IAM permissions for CodeBuild role

### Website Not Loading

1. Check S3 bucket policy allows public access
2. Verify static website hosting is enabled
3. Check bucket region matches your configuration

## Cost Considerations

- **CodeBuild**: ~$0.005 per build minute (Large instance)
- **Lambda**: Pay per request + compute time (first 1M requests free)
- **ECR**: Storage costs for Docker images (~$0.10 per GB/month)
- **S3**: Storage + requests costs (minimal for static hosting)

## Architecture

```
┌─────────────┐
│  CodeBuild  │ → Builds Docker image
└──────┬──────┘
       │
       ↓
┌─────────────┐
│     ECR     │ → Stores Docker image
└──────┬──────┘
       │
       ↓
┌─────────────┐      ┌──────────────┐
│   Lambda    │ ←──→ │ Function URL │
└──────┬──────┘      └──────────────┘
       │
       ↓
┌─────────────┐
│     S3      │ → Hosts website
└─────────────┘
```
