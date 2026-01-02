#!/bin/bash
# Deploy the test web page to S3

set -e

BUCKET_NAME=${1:-"california-housing-predictor"}
REGION=${AWS_REGION:-us-east-1}

echo "🌐 Deploying website to S3..."

# Create bucket if it doesn't exist
if ! aws s3 ls "s3://${BUCKET_NAME}" 2>/dev/null; then
    echo "📦 Creating S3 bucket: ${BUCKET_NAME}"
    if [ "$REGION" == "us-east-1" ]; then
        aws s3 mb "s3://${BUCKET_NAME}"
    else
        aws s3 mb "s3://${BUCKET_NAME}" --region "${REGION}"
    fi
fi

# Upload index.html
echo "📤 Uploading index.html..."
aws s3 cp aws-deployment/index.html "s3://${BUCKET_NAME}/index.html" --content-type "text/html"

# Enable static website hosting
echo "⚙️  Configuring static website hosting..."
aws s3 website "s3://${BUCKET_NAME}/" \
    --index-document index.html \
    --error-document index.html

# Set bucket policy for public read access
echo "🔓 Setting bucket policy for public access..."
cat > /tmp/bucket-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::${BUCKET_NAME}/*"
    }
  ]
}
EOF

aws s3api put-bucket-policy \
    --bucket "${BUCKET_NAME}" \
    --policy file:///tmp/bucket-policy.json

# Get website URL
if [ "$REGION" == "us-east-1" ]; then
    WEBSITE_URL="http://${BUCKET_NAME}.s3-website-${REGION}.amazonaws.com"
else
    WEBSITE_URL="http://${BUCKET_NAME}.s3-website.${REGION}.amazonaws.com"
fi

echo ""
echo "✅ Website deployed!"
echo "🌐 Website URL: ${WEBSITE_URL}"
echo ""
echo "📝 Note: Update the Lambda Function URL in the web page after deploying Lambda."


