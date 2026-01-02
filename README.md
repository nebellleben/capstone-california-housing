# 🎓 Capstone Project 1 — California Housing Prediction

Capstone project for Machine Learning Zoomcamp Cohort 2025

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Model Performance Discussion](#-model-performance-discussion)
- [Usage](#usage)

## 🎯 Project Overview

### Problem Statement

This project aims to predict median house values in California using various machine learning techniques. The goal is to compare different regression models including baseline linear regression, regularized models (Lasso and Ridge), and deep learning approaches using Keras/TensorFlow.

### Objectives

- Perform comprehensive exploratory data analysis (EDA) on California housing data
- Implement and compare multiple regression models:
  - Linear Regression (Baseline)
  - Lasso Regression with hyperparameter tuning
  - Ridge Regression with hyperparameter tuning
  - Neural Network using Keras/TensorFlow
- Evaluate model performance using RMSE (Root Mean Squared Error)
- Gain familiarity with regression, deep learning, and containerization

## 📊 Dataset

The California Housing dataset contains information about housing districts in California with the following features:

- **longitude**: Longitude of the district
- **latitude**: Latitude of the district
- **housing_median_age**: Median age of houses in the district
- **total_rooms**: Total number of rooms in the district
- **total_bedrooms**: Total number of bedrooms in the district
- **population**: Population of the district
- **households**: Number of households in the district
- **median_income**: Median income of households in the district
- **median_house_value**: Median house value (target variable)
- **ocean_proximity**: Categorical variable indicating proximity to ocean

**Dataset Statistics:**
- Total samples: 20,640
- Features: 10 (9 input features + 1 target)
- Missing values: 207 in `total_bedrooms` column

## 🚀 Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd capstone-california-housing
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

4. Install Jupyter kernel (for notebook support):
```bash
uv sync --dev  # Install dev dependencies including Jupyter
uv run python -m ipykernel install --user --name=capstone-california-housing --display-name "Python (uv: capstone-california-housing)"
```

The kernel is now registered and available in Jupyter. When opening a notebook, select **"Python (uv: capstone-california-housing)"** as the kernel to use the `uv` environment.

## 📁 Project Structure

```
capstone-california-housing/
├── project.ipynb          # Main Jupyter notebook with EDA and model training
├── housing.csv            # California housing dataset
├── pyproject.toml         # Project dependencies (uv)
├── requirements.txt       # Alternative requirements file
├── uv.lock               # Locked dependencies
├── train.py              # Training script (extracted from notebook)
├── predict.py            # FastAPI prediction server
├── main.py               # Main entry point (placeholder)
├── generate_eda_plots.py # Script to generate EDA visualization images
├── model.pkl             # Trained models (DictVectorizer + 4 models)
├── images/               # Generated EDA plots
│   ├── eda_histograms.png
│   ├── eda_geographic.png
│   ├── eda_correlation.png
│   ├── eda_ocean_proximity.png
│   └── eda_boxplot_ocean.png
├── Dockerfile            # Containerization setup for prediction server
└── README.md             # This file
```

## 📈 Exploratory Data Analysis (EDA)

### Data Overview

The dataset contains 20,640 housing districts with 10 features. Key observations:

- **Numerical Features**: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income
- **Categorical Features**: ocean_proximity
- **Target Variable**: median_house_value

### EDA Visualizations

The following visualizations provide insights into the dataset. To regenerate these plots, run:
```bash
uv run python generate_eda_plots.py
```

#### 1. Histograms of Numerical Features

![Histograms of Numerical Features](images/eda_histograms.png)

**Key Insights:**
- Most features show right-skewed distributions
- `median_income` appears to be capped at around 15
- `total_rooms`, `total_bedrooms`, `population`, and `households` have long tails
- Geographic features (`longitude`, `latitude`) show normal-like distributions

#### 2. Geographic Distribution

![Geographic Distribution of House Values](images/eda_geographic.png)

**Key Insights:**
- Higher house values are concentrated near the coast (especially around San Francisco Bay Area and Los Angeles)
- Inland areas generally show lower median house values
- Clear geographic patterns in pricing
- Coastal regions show significantly higher median house values

#### 3. Correlation Matrix

![Correlation Matrix](images/eda_correlation.png)

**Key Insights:**
- `total_rooms`, `total_bedrooms`, `population`, and `households` are highly correlated (0.8+)
- `median_income` shows the strongest correlation with `median_house_value` (~0.69)
- `housing_median_age` has weak correlations with other features
- Geographic features (`longitude`, `latitude`) show moderate correlations with house values

#### 4. Ocean Proximity Distribution

![Ocean Proximity Distribution](images/eda_ocean_proximity.png)

**Key Insights:**
- Most districts are located inland or near the ocean
- Island districts are extremely rare
- Distribution is imbalanced across categories

#### 5. House Value by Ocean Proximity

![House Value by Ocean Proximity](images/eda_boxplot_ocean.png)

**Key Insights:**
- ISLAND category shows the highest median house values (though with very few samples)
- NEAR BAY and NEAR OCEAN show higher values than INLAND
- Clear relationship between ocean proximity and house values

### Data Preprocessing

1. **Data Type Conversion:**
   - Converted `ocean_proximity` to categorical
   - Converted integer features (`housing_median_age`, `total_rooms`, `households`, `population`) to int64

2. **Train/Validation/Test Split:**
   - Training set: 60% (12,384 samples)
   - Validation set: 20% (4,128 samples)
   - Test set: 20% (4,128 samples)
   - Random state: 42 for reproducibility

3. **Feature Encoding:**
   - Used `DictVectorizer` to encode categorical and numerical features
   - Sparse matrix representation for memory efficiency

## 🤖 Model Training and Evaluation

### Models Implemented

1. **Linear Regression (Baseline)**
   - Simple linear regression as a baseline model
   - No regularization

2. **Lasso Regression**
   - L1 regularization for feature selection
   - Hyperparameter tuning: alpha values from 0.001 to 1000
   - Best alpha: 24 (lowest validation RMSE)

3. **Ridge Regression**
   - L2 regularization to prevent overfitting
   - Hyperparameter tuning: alpha values from 0.001 to 1000
   - Best alpha: 10 (lowest validation RMSE)

4. **Neural Network (Keras/TensorFlow)**
   - Architecture:
     - Input layer: 13 features (after encoding)
     - Hidden layer 1: 64 neurons with ReLU activation
     - Hidden layer 2: 64 neurons with ReLU activation
     - Output layer: 1 neuron (regression)
   - Optimizer: Adam with gradient clipping (clipnorm=1.0)
   - Loss function: Mean Squared Error
   - Metrics: Root Mean Squared Error
   - Training: 200 epochs, batch size 32
   - Performance: Best model with Validation RMSE: 66,436.45 and Test RMSE: 66,156.31

### Model Comparison Visualization

```python
# Plot predictions vs actual values for all models
plt.figure(figsize=(10, 6))
plt.scatter(y_val, baseline_predictions, alpha=0.5, label='Baseline Model')
plt.scatter(y_val, lasso_predictions, alpha=0.5, label='Lasso Model')
plt.scatter(y_val, ridge_predictions, alpha=0.5, label='Ridge Model')
plt.scatter(y_val, nn_predictions, alpha=0.5, label='Neural Network')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
         'k--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Model Predictions Comparison')
plt.legend()
plt.show()
```

## 📊 Results

### Validation Set Performance

| Model | RMSE | Notes |
|-------|------|-------|
| **Linear Regression (Baseline)** | 68,645.28 | Baseline model |
| **Lasso Regression** | 68,564.26 | Best alpha: 24 |
| **Ridge Regression** | 68,569.34 | Best alpha: 10 |
| **Neural Network** | 66,436.45 | 200 epochs, 64-64 architecture |

### Test Set Performance

| Model | RMSE | Notes |
|-------|------|-------|
| **Linear Regression (Baseline)** | 71,165.85 | Baseline model |
| **Lasso Regression** | 71,222.11 | Best alpha: 24 |
| **Ridge Regression** | 71,243.57 | Best alpha: 10 |
| **Neural Network** | 66,156.31 | 200 epochs, 64-64 architecture |

### Key Findings

1. **Neural Network** achieved the best performance on both validation and test sets
2. **Lasso Regression** performed best among linear models on the validation set
3. **Linear Regression (Baseline)** performed best among linear models on the test set
4. Regularized models (Lasso and Ridge) performed slightly better than the baseline on validation set
5. Neural network showed significantly better performance (~5,000-6,000 RMSE improvement) compared to linear models
6. All linear models showed similar performance, suggesting the problem may benefit from:
   - Feature engineering
   - More complex neural network architectures
   - Ensemble methods

## 🔍 Model Performance Discussion

### Overall Performance Analysis

All four models demonstrated remarkably similar performance, with RMSE values clustering around 68,000-69,000. This convergence suggests that:

1. **Linear relationships dominate**: The strong linear relationships in the data (particularly with `median_income`) make linear models highly effective
2. **Limited non-linearity**: The dataset may not contain significant non-linear patterns that would give neural networks a substantial advantage
3. **Feature quality**: The engineered features (geographic coordinates, derived ratios) are well-suited for linear regression approaches

### Model-by-Model Analysis

#### 1. Linear Regression (Baseline)
- **Performance**: Serves as a strong baseline with RMSE ~68,581
- **Strengths**: 
  - Highly interpretable coefficients
  - Fast training and prediction
  - No hyperparameter tuning required
  - Provides feature importance insights
- **Weaknesses**: 
  - No regularization, potentially more sensitive to outliers
  - Cannot capture complex interactions without feature engineering
- **Use Case**: Best for scenarios requiring interpretability and when model simplicity is valued

#### 2. Lasso Regression
- **Performance**: Slightly better than baseline (RMSE ~68,564) with optimal alpha=24
- **Strengths**:
  - L1 regularization performs automatic feature selection
  - Reduces overfitting risk
  - Can zero out less important features
  - Maintains interpretability
- **Weaknesses**:
  - Requires hyperparameter tuning
  - May be too aggressive in feature selection for this dataset
- **Use Case**: When feature selection is desired or when dealing with high-dimensional data

#### 3. Ridge Regression
- **Performance**: Competitive linear model (Validation RMSE: 68,569.34, Test RMSE: 71,243.57) with optimal alpha=10
- **Strengths**:
  - L2 regularization prevents overfitting while retaining all features
  - Better generalization than baseline on validation set
  - More stable predictions than Lasso
  - Still interpretable
- **Weaknesses**:
  - Requires hyperparameter tuning
  - Does not perform feature selection
- **Use Case**: Good choice when interpretability is important and slight performance trade-off is acceptable

#### 4. Neural Network
- **Performance**: Best performing model (Validation RMSE: 66,436.45, Test RMSE: 66,156.31) with 64-64 architecture
- **Strengths**:
  - Captures non-linear relationships and feature interactions effectively
  - Significantly better performance than linear models (~5,000-5,500 RMSE improvement)
  - Flexible architecture allows for complex pattern learning
  - Potential for further improvement with more sophisticated architectures
- **Weaknesses**:
  - Requires more computational resources
  - Longer training time (200 epochs)
  - Less interpretable (black box)
  - Sensitive to hyperparameters and initialization
  - Risk of overfitting without proper regularization
- **Use Case**: **Recommended for production** - best performance when interpretability is less critical

### Key Insights

#### Why Neural Network Performed Best
1. **Non-linear Patterns**: The neural network successfully captured non-linear relationships and feature interactions that linear models cannot
2. **Architecture Effectiveness**: The 64-64 architecture with ReLU activations and gradient clipping proved effective for this problem
3. **Generalization**: The model showed good generalization with test RMSE (66,156.31) close to validation RMSE (66,436.45)

#### Why Linear Models Showed Similar Performance
1. **Linear Dominance**: The problem has strong linear components, with `median_income` showing strong linear correlation (~0.69)
2. **Limited Non-linearity**: While non-linear patterns exist, they are not dominant enough to give neural networks a massive advantage
3. **Feature Quality**: The engineered features work well for linear regression approaches

#### Performance Patterns
The results show:
- **Neural Network Advantage**: Neural network achieved ~5,000-5,500 RMSE improvement over linear models, demonstrating the value of capturing non-linear patterns
- **Linear Model Convergence**: All linear models (Baseline, Lasso, Ridge) showed similar performance, suggesting regularization provides marginal benefits for this dataset
- **Data Quality**: The features are well-engineered and informative, allowing both linear and non-linear models to perform reasonably well

### Practical Recommendations

#### For Production Deployment
1. **Primary Choice**: **Neural Network** - Best performance (Test RMSE: 66,156.31) when interpretability is less critical
2. **Alternative**: **Linear Regression (Baseline)** - Best among linear models on test set (Test RMSE: 71,165.85) with full interpretability
3. **Ensemble Approach**: Consider averaging predictions from Neural Network and best linear model for potentially better generalization

#### For Further Improvement
1. **Feature Engineering**:
   - Create interaction terms (e.g., `median_income × ocean_proximity`)
   - Derive new features (e.g., rooms per household, bedrooms per room)
   - Geographic clustering or distance-based features
2. **Model Enhancements**:
   - Try ensemble methods (Random Forest, Gradient Boosting)
   - Experiment with deeper/wider neural networks
   - Apply stacking or blending techniques
3. **Data Improvements**:
   - Collect more features (school quality, crime rates, etc.)
   - Handle missing values more sophisticatedly
   - Consider temporal features if data spans time

### Limitations

1. **RMSE Interpretation**: An RMSE of ~68,000 means predictions are off by approximately $68,000 on average, which is significant given median house values around $200,000-300,000
2. **Feature Limitations**: The dataset may lack important predictive features (e.g., property condition, neighborhood amenities)
3. **Geographic Granularity**: District-level aggregation may mask important local variations
4. **Temporal Aspects**: The dataset is a snapshot; market conditions change over time

### Conclusion

The Neural Network achieved the best performance with a significant improvement (~5,000-5,500 RMSE) over linear models. However, **model selection should be based on practical considerations** (interpretability, deployment complexity, inference speed) rather than performance alone. 

- For **maximum performance**: Use the Neural Network (Test RMSE: 66,156.31)
- For **interpretability and simplicity**: Use Linear Regression (Test RMSE: 71,165.85)
- For **balanced approach**: Consider an ensemble of Neural Network and Linear Regression

The choice depends on whether the ~5,000 RMSE improvement justifies the added complexity and reduced interpretability of the neural network.

## 💻 Usage

### Running the Notebook

1. Start Jupyter:
```bash
uv run jupyter notebook
# or
uv run jupyter lab
```

2. Open `project.ipynb` and select the kernel **"Python (uv: capstone-california-housing)"** from the kernel menu

3. Run all cells

**Note**: The kernel uses the `uv` virtual environment (`.venv`) automatically, so all packages installed via `uv` are available in the notebook.

### Running Training Scripts

```bash
# Train models and save to model.pkl
uv run python train.py
```

### Running the Prediction Server

The project includes a FastAPI-based prediction server that provides a REST API for making predictions.

#### Start the Server

```bash
# Make sure model.pkl exists (run train.py first if needed)
uv run python predict.py
```

The server will start on `http://localhost:9696`

#### Using the API

1. **Interactive API Documentation**: Visit `http://localhost:9696/docs` for Swagger UI documentation

2. **Make Predictions**: Send POST requests to `/predict` endpoint:

```bash
curl -X POST "http://localhost:9696/predict" \
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

**Response Format:**
```json
{
  "median house value (baseline model)": 410292.95,
  "median house value (lasso model)": 410470.09,
  "median house value (ridge model)": 410279.73,
  "median house value (neural network model)": 398736.84
}
```

**Required Input Fields:**
- `longitude`: Longitude coordinate (float)
- `latitude`: Latitude coordinate (float)
- `housing_median_age`: Median age of houses (float)
- `total_rooms`: Total number of rooms (float)
- `total_bedrooms`: Total number of bedrooms (float)
- `population`: Population count (float)
- `households`: Number of households (float)
- `median_income`: Median income (float)
- `ocean_proximity`: One of "NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND" (string)

### Using Docker

The Dockerfile is configured to run the prediction server:

```bash
# Build the Docker image
docker build -t california-housing .

# Run the container (exposes port 9696)
docker run -p 9696:9696 california-housing
```

The prediction API will be available at `http://localhost:9696`

## 🔧 Dependencies

Main dependencies (managed via `uv`):

- **pandas** >= 2.3.3 - Data manipulation
- **numpy** >= 2.0.2 - Numerical computing
- **matplotlib** >= 3.9.4 - Plotting
- **seaborn** >= 0.13.2 - Statistical visualization
- **scikit-learn** >= 1.6.1 - Machine learning models
- **tensorflow** >= 2.20.0 - Deep learning framework
- **fastapi** >= 0.128.0 - Web framework for prediction API
- **uvicorn** >= 0.39.0 - ASGI server for FastAPI

See `pyproject.toml` or `requirements.txt` for complete dependency list.

## 📝 Notes

- The project uses `uv` for fast package management
- All models (DictVectorizer + 4 models) are saved to `model.pkl` using pickle
- Random seeds are set for reproducibility (random_state=42)
- Missing values in `total_bedrooms` are handled by filling with median from training set
- The prediction server loads all models and returns predictions from all four models
- Models are trained on 60% of data, validated on 20%, and tested on 20%

## 🎓 Learning Outcomes

This project demonstrates:
- Comprehensive EDA techniques
- Multiple regression model implementations
- Hyperparameter tuning
- Deep learning with Keras/TensorFlow
- Model evaluation and comparison
- REST API development with FastAPI
- Containerization with Docker
- Model deployment and serving

## 📄 License

This project is part of the Machine Learning Zoomcamp Cohort 2025.

---

**Author**: Kelvin Chan  
**Course**: Machine Learning Zoomcamp Cohort 2025  
**Project Type**: Capstone Project 1
