#!/bin/bash
# Complete fix for Lambda Function URL 403 Forbidden Error
# This script will delete and recreate the Function URL with proper permissions

set -e

REGION=${AWS_REGION:-us-east-1}
FUNCTION_NAME="california-housing-predict"

echo "🔧 Complete Fix for Lambda Function URL 403 Error"
echo "=================================================="
echo ""

# Step 1: Check if function exists
echo "📋 Step 1: Checking Lambda function..."
if ! aws lambda get-function --function-name ${FUNCTION_NAME} --region ${REGION} > /dev/null 2>&1; then
    echo "❌ Error: Function ${FUNCTION_NAME} not found"
    exit 1
fi
echo "✅ Function exists"

# Step 2: Delete existing Function URL (if exists)
echo ""
echo "📋 Step 2: Removing existing Function URL (if any)..."
aws lambda delete-function-url-config \
    --function-name ${FUNCTION_NAME} \
    --region ${REGION} 2>/dev/null && echo "✅ Old Function URL deleted" || echo "ℹ️  No existing Function URL to delete"

# Wait a moment for deletion to propagate
sleep 2

# Step 3: Remove any existing permissions
echo ""
echo "📋 Step 3: Cleaning up existing permissions..."
aws lambda remove-permission \
    --function-name ${FUNCTION_NAME} \
    --statement-id FunctionURLAllowPublicAccess \
    --region ${REGION} 2>/dev/null && echo "✅ Old permission removed" || echo "ℹ️  No existing permission to remove"

sleep 1

# Step 4: Create new Function URL with NONE auth
echo ""
echo "📋 Step 4: Creating new Function URL with public access..."
FUNCTION_URL=$(aws lambda create-function-url-config \
    --function-name ${FUNCTION_NAME} \
    --auth-type NONE \
    --cors '{"AllowOrigins":["*"],"AllowMethods":["*"],"AllowHeaders":["*"],"ExposeHeaders":["*"],"MaxAge":86400}' \
    --region ${REGION} \
    --query 'FunctionUrl' --output text)

if [ -z "$FUNCTION_URL" ]; then
    echo "❌ Error: Failed to create Function URL"
    exit 1
fi

echo "✅ Function URL created: ${FUNCTION_URL}"

# Step 5: Explicitly add public access permission
echo ""
echo "📋 Step 5: Adding explicit public access permission..."
aws lambda add-permission \
    --function-name ${FUNCTION_NAME} \
    --statement-id FunctionURLAllowPublicAccess \
    --action lambda:InvokeFunctionUrl \
    --principal "*" \
    --function-url-auth-type NONE \
    --region ${REGION} 2>&1 | grep -E "(already exists|Statement)" || echo "✅ Permission added"

# Step 6: Verify configuration
echo ""
echo "📋 Step 6: Verifying configuration..."
CONFIG=$(aws lambda get-function-url-config \
    --function-name ${FUNCTION_NAME} \
    --region ${REGION})

AUTH_TYPE=$(echo "$CONFIG" | grep -o '"AuthType": "[^"]*"' | cut -d'"' -f4)
echo "   Auth Type: ${AUTH_TYPE}"

# Step 7: Test the Function URL
echo ""
echo "📋 Step 7: Testing Function URL..."
echo ""

echo "Test 1: Health check (GET /)"
HEALTH_RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X GET "${FUNCTION_URL}/")
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | grep "HTTP_CODE" | cut -d: -f2)
BODY=$(echo "$HEALTH_RESPONSE" | sed '/HTTP_CODE/d')

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ Health check passed (HTTP $HTTP_CODE)"
    echo "   Response: $BODY"
else
    echo "❌ Health check failed (HTTP $HTTP_CODE)"
    echo "   Response: $BODY"
fi

echo ""
echo "Test 2: Prediction (POST /predict)"
PREDICT_RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X POST "${FUNCTION_URL}/predict" \
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
    }')
HTTP_CODE=$(echo "$PREDICT_RESPONSE" | grep "HTTP_CODE" | cut -d: -f2)
BODY=$(echo "$PREDICT_RESPONSE" | sed '/HTTP_CODE/d')

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ Prediction test passed (HTTP $HTTP_CODE)"
    echo "   Response preview: $(echo "$BODY" | head -c 100)..."
else
    echo "❌ Prediction test failed (HTTP $HTTP_CODE)"
    echo "   Response: $BODY"
fi

# Summary
echo ""
echo "=================================================="
echo "✅ Function URL Fix Complete!"
echo ""
echo "🌐 Function URL: ${FUNCTION_URL}"
echo ""
echo "📝 Next steps:"
echo "   1. Update your website with this URL: ${FUNCTION_URL}"
echo "   2. Test the website at: http://california-housing-predictor.s3-website-us-east-1.amazonaws.com"
echo ""
echo "🧪 Manual test commands:"
echo "   curl -X GET \"${FUNCTION_URL}/\""
echo "   curl -X POST \"${FUNCTION_URL}/predict\" -H \"Content-Type: application/json\" -d '{\"longitude\":-122.23,\"latitude\":37.88,\"housing_median_age\":41.0,\"total_rooms\":880.0,\"total_bedrooms\":129.0,\"population\":322.0,\"households\":126.0,\"median_income\":8.3252,\"ocean_proximity\":\"NEAR BAY\"}'"

