# Machine Learning Development Life Cycle (MLDLC)

**Complete Guide with Real-World Examples and Code**

---

## Table of Contents

1. Phase 1: Framing the Problem
2. Phase 2: Gathering Data
3. Phase 3: Data Processing & Cleaning
4. Phase 4: Exploratory Data Analysis (EDA)
5. Phase 5: Feature Engineering & Selection
6. Phase 6: Model Training, Evaluation & Selection
7. Phase 7: Model Deployment
8. Phase 8: Testing in Production
9. Phase 9: Optimization & Continuous Improvement

---

## Phase 1: Framing the Problem

### What is Problem Framing?

Problem framing is the most critical phase. A well-defined problem leads to success; a poorly-defined problem leads to wasted resources.

### Key Components to Define

**1. Business Goal**

The ultimate objective the organization wants to achieve.

Example: "Reduce customer churn rate from 15% to 10% within 6 months"

```
Not: "Build a churn prediction model"
But: "Identify customers likely to churn so we can offer retention incentives, 
      reducing churn by 5% and saving $2M annually"
```

**2. Problem Type**

Classify the ML problem into one of these categories:

- **Classification**: Predicting a categorical label
  - Binary: Fraud/Not Fraud, Churn/No Churn
  - Multi-class: Product Category A/B/C, Risk Level (Low/Medium/High)
  
- **Regression**: Predicting continuous numeric values
  - House prices, stock prices, demand forecast
  
- **Ranking/Recommendation**: Ordering items by relevance
  - Product recommendations, search ranking
  
- **Clustering**: Grouping similar items
  - Customer segmentation, anomaly detection
  
- **Time Series**: Predicting future values based on temporal patterns
  - Stock price forecasting, energy demand prediction

```
Example: Churn prediction is a Classification problem (Binary)
```

**3. Success Metrics (Business & Technical)**

Business metrics measure impact on organization:
- Revenue increase, cost reduction, customer retention
- Customer satisfaction, risk reduction

Technical metrics measure model performance:
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Regression: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), R²
- Ranking: NDCG, MAP (Mean Average Precision)

```
Example for Churn Prediction:

Business Metric: 
- Reduce churn by 5% = $2M annual savings
- Cost per false positive (incorrect prediction): $50 (wrong incentive sent)
- Cost per false negative (missed churn): $500 (lost customer)

Technical Metric:
- Precision >= 80% (don't waste incentives on non-churners)
- Recall >= 70% (catch most actual churners)
- Minimum F1 score: 0.75
```

**4. Constraints & Requirements**

- **Latency**: How fast must predictions be made?
  - Real-time (< 100ms): Fraud detection, autonomous vehicles
  - Batch (daily/weekly): Demand forecasting, campaign targeting
  
- **Interpretability**: Must we explain why the model made a prediction?
  - High need: Healthcare, finance, legal (regulatory compliance)
  - Low need: Recommendation systems, content ranking
  
- **Scale**: How many predictions per day?
  - Millions/billions: Social media, finance
  - Thousands: Internal business use
  
- **Data availability**: How much quality data do we have?
  - Rich data: Easy
  - Limited labeled data: Need semi-supervised or transfer learning
  
- **Fairness/Bias**: Are there protected attributes we must avoid?
  - Example: Credit scoring cannot discriminate by race/gender

```
Example Problem Frame (Complete):

Project: Customer Churn Prediction

Business Goal:
- Identify customers likely to churn in next 30 days
- Enable retention team to target with personalized offers
- Target: Reduce churn from 15% to 10%, save $2M annually

Problem Type: Binary Classification

Success Metrics:
- Technical: Precision >= 80%, Recall >= 70%, F1 >= 0.75, ROC-AUC >= 0.85
- Business: 5% absolute churn reduction, $2M cost savings, positive ROI

Constraints:
- Latency: Batch processing (daily/weekly acceptable)
- Interpretability: Medium (explain key drivers, no need for perfect explainability)
- Scale: 10M customers, 1M predictions per run
- Data: 2 years of transaction data, ~50K historical churners
- Fairness: Avoid bias by customer demographics
```

---

## Phase 2: Gathering Data

### What Data is Needed?

Once the problem is framed, identify all relevant data sources.

### Data Sources

**Internal Sources**

1. **Transactional Data**
   - Database tables with customer transactions
   - Tools: SQL, PostgreSQL, MySQL, BigQuery, Snowflake
   
   ```sql
   SELECT customer_id, transaction_amount, transaction_date, product_category
   FROM transactions
   WHERE transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 24 MONTH)
   ```

2. **Customer Data**
   - CRM systems (Salesforce, HubSpot)
   - Contains: demographics, contact info, communication history
   
3. **Log Data**
   - Website/app activity logs
   - Contains: page views, clicks, session duration
   - Tools: Apache Kafka, Splunk, CloudWatch

**External Sources**

1. **Public Datasets**
   - Kaggle, UCI Machine Learning Repository, OpenML
   - Free but usually for learning/competition
   
2. **APIs**
   - Weather data (OpenWeatherMap)
   - Financial data (Alpha Vantage, IEX Cloud)
   - Market data (Yahoo Finance)
   
3. **Web Scraping** (with legal/ethical approval)
   - Tools: BeautifulSoup, Selenium, Scrapy
   - Must respect ToS and robots.txt
   
4. **Third-party Data Providers**
   - Demographic data, credit scores, market data
   - Providers: Experian, Equifax, Bloomberg

### Data Requirements Checklist

```
Frame the Problem checklist:

[✓] Does the data contain the prediction TARGET?
    Example: Churn status (yes/no) for historical customers

[✓] Does the data cover sufficient time period?
    Example: At least 12-24 months for seasonality patterns

[✓] Is the volume adequate?
    Example: Need >1000 positive examples for binary classification

[✓] Are there privacy/compliance concerns?
    Example: PII (personally identifiable information), GDPR, HIPAA

[✓] Is the data quality acceptable?
    Example: <10% missing values, no obvious errors

[✓] Will the data be available at prediction time?
    Example: Don't use future data to predict the past
```

### Example: Gathering Data for Churn Prediction

```python
import pandas as pd
import numpy as np

# Query 1: Get customer transactions (18-24 months)
query_transactions = """
SELECT 
    customer_id,
    SUM(transaction_amount) as total_spent,
    COUNT(*) as transaction_count,
    MAX(transaction_date) as last_transaction_date,
    MIN(transaction_date) as first_transaction_date
FROM transactions
WHERE transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 24 MONTH)
GROUP BY customer_id
"""

# Query 2: Get customer demographics
query_demographics = """
SELECT 
    customer_id,
    age,
    gender,
    account_creation_date,
    account_type,
    country
FROM customers
"""

# Query 3: Get support tickets (engagement signal)
query_support = """
SELECT 
    customer_id,
    COUNT(*) as support_tickets,
    MAX(ticket_date) as last_support_date
FROM support_tickets
WHERE ticket_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 24 MONTH)
GROUP BY customer_id
"""

# Query 4: Get target variable (churn)
query_churn = """
SELECT 
    customer_id,
    CASE WHEN last_purchase_date < DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) 
         THEN 1 ELSE 0 END as churned
FROM customers
"""

# Combine all data
transactions_df = pd.read_sql(query_transactions, connection)
demographics_df = pd.read_sql(query_demographics, connection)
support_df = pd.read_sql(query_support, connection)
churn_df = pd.read_sql(query_churn, connection)

# Merge all tables
raw_data = transactions_df.merge(demographics_df, on='customer_id')
raw_data = raw_data.merge(support_df, on='customer_id', how='left')
raw_data = raw_data.merge(churn_df, on='customer_id')

print(f"Dataset shape: {raw_data.shape}")
print(f"Churn rate: {raw_data['churned'].mean():.2%}")
```

### Tools Used in Data Collection

| Tool | Purpose | Use Case |
|------|---------|----------|
| SQL / BigQuery | Query databases | Extract transactional data |
| Apache Spark | Large-scale data processing | Process 100GB+ datasets |
| Pandas | Data manipulation in Python | Small-medium datasets |
| Kafka | Stream data collection | Real-time data pipelines |
| Scrapy / BeautifulSoup | Web scraping | Collect web data |
| APIs | Access third-party data | Weather, financial data |
| AWS S3 / GCS | Cloud storage | Store raw data files |

---

## Phase 3: Data Processing & Cleaning

### Why is Data Cleaning Critical?

"Garbage in, garbage out" - Data quality directly impacts model quality.

According to research, data scientists spend 70-80% of time on data cleaning and preparation.

### Common Data Quality Issues

**1. Missing Values**

```python
import pandas as pd
import numpy as np

# Load data
df = raw_data.copy()

# Check missing values
print(df.isnull().sum())
print(df.isnull().sum() / len(df) * 100)  # Percentage

# Output example:
# transaction_count    0 (0.0%)
# total_spent          0 (0.0%)
# age                  542 (5.4%)
# support_tickets      2104 (21%)
# last_support_date    2104 (21%)
```

**Strategies for Handling Missing Values:**

```python
# Strategy 1: Drop rows (only if <5% missing)
df_dropped = df.dropna(subset=['age'])

# Strategy 2: Drop columns (only if >50% missing)
df = df.drop(columns=['last_support_date'])  # 21% missing, might drop

# Strategy 3: Imputation with mean/median (numeric)
df['age'].fillna(df['age'].median(), inplace=True)

# Strategy 4: Imputation with mode (categorical)
df['country'].fillna(df['country'].mode()[0], inplace=True)

# Strategy 5: Forward fill / Backward fill (time series)
df['balance'] = df['balance'].fillna(method='ffill')

# Strategy 6: Advanced imputation (ML-based)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
```

**2. Duplicate Records**

```python
# Check duplicates
print(df.duplicated().sum())  # Exact duplicates

# Check duplicates on specific columns
print(df.duplicated(subset=['customer_id']).sum())

# Remove duplicates (keep first occurrence)
df = df.drop_duplicates(subset=['customer_id'], keep='first')
```

**3. Data Type Issues**

```python
# Identify type issues
print(df.dtypes)

# Example: Age column stored as string "25" instead of int
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Convert dates
df['account_creation_date'] = pd.to_datetime(df['account_creation_date'], 
                                             errors='coerce')

# Convert categorical
df['account_type'] = df['account_type'].astype('category')
```

**4. Inconsistent/Invalid Values**

```python
# Check ranges
print(df['age'].describe())

# Example: Age > 150 is unrealistic
df = df[df['age'] <= 120]

# Example: Transaction amount < 0 is invalid
df = df[df['transaction_amount'] > 0]

# Case inconsistency in categorical
df['country'] = df['country'].str.upper()

# Whitespace issues
df['country'] = df['country'].str.strip()
```

**5. Outliers**

```python
# Method 1: Statistical (IQR)
Q1 = df['transaction_amount'].quantile(0.25)
Q3 = df['transaction_amount'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['transaction_amount'] < Q1 - 1.5*IQR) | 
              (df['transaction_amount'] > Q3 + 1.5*IQR)]

# Method 2: Z-score
from scipy import stats
z_scores = np.abs(stats.zscore(df['transaction_amount']))
outliers = df[z_scores > 3]

# Method 3: Isolation Forest (ML-based anomaly detection)
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.05)
outliers = df[iso_forest.fit_predict(df[['transaction_amount']]) == -1]

# Decision: Keep or Remove? 
# For fraud detection: KEEP (outliers might be fraudulent)
# For customer churn: REMOVE (extreme cases might distort pattern)
```

### Complete Data Cleaning Pipeline

```python
def clean_data(df):
    """
    Complete data cleaning pipeline
    """
    # Step 1: Remove duplicates
    df = df.drop_duplicates(subset=['customer_id'], keep='first')
    
    # Step 2: Type conversion
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['account_creation_date'] = pd.to_datetime(df['account_creation_date'])
    
    # Step 3: Handle missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna('Unknown', inplace=True)
    
    # Step 4: Handle outliers (remove extreme cases)
    df = df[df['age'] <= 120]
    df = df[df['transaction_amount'] > 0]
    
    # Step 5: Standardize formats
    df['country'] = df['country'].str.upper().str.strip()
    
    # Step 6: Log cleaning report
    print(f"Final dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df

cleaned_data = clean_data(raw_data)
```

### Tools Used in Data Cleaning

| Tool | Purpose |
|------|---------|
| Pandas | Data manipulation, cleaning |
| NumPy | Numerical operations |
| Great Expectations | Data validation & profiling |
| Deequ | Large-scale data quality |
| OpenRefine | Visual data cleaning |
| Apache Spark | Distributed data cleaning |

---

## Phase 4: Exploratory Data Analysis (EDA)

### What is EDA?

EDA is the process of understanding data through visualization and statistical analysis BEFORE building models.

### EDA Goals

1. Understand feature distributions
2. Identify relationships between features and target
3. Detect data leakage and inconsistencies
4. Guide feature engineering decisions
5. Communicate insights to stakeholders

### Key EDA Techniques

**1. Univariate Analysis (Single Variable)**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Numeric variable: Distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram
df['age'].hist(bins=30, ax=axes[0])
axes[0].set_title('Age Distribution (Histogram)')
axes[0].set_xlabel('Age')

# KDE plot (smooth distribution)
df['age'].plot(kind='kde', ax=axes[1])
axes[1].set_title('Age Distribution (KDE)')

# Box plot (shows outliers)
df['age'].plot(kind='box', ax=axes[2])
axes[2].set_title('Age Distribution (Box Plot)')

plt.tight_layout()
plt.show()

# Statistical summary
print(df['age'].describe())
# Output:
# count    10000
# mean      42.5
# std       15.2
# min       18
# 25%       32
# 50%       42
# 75%       53
# max       120
```

**2. Categorical Variable Distribution**

```python
# Count plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df['account_type'].value_counts().plot(kind='bar', ax=axes[0])
axes[0].set_title('Account Type Distribution')

# Percentage
(df['account_type'].value_counts() / len(df) * 100).plot(
    kind='pie', ax=axes[1], autopct='%1.1f%%'
)
axes[1].set_title('Account Type Percentage')

plt.tight_layout()
plt.show()
```

**3. Bivariate Analysis (Two Variables)**

```python
# Numeric vs Numeric: Scatter plot & Correlation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Scatter plot
axes[0].scatter(df['age'], df['total_spent'], alpha=0.5)
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Total Spent ($)')
axes[0].set_title('Age vs Total Spent')

# Correlation
correlation = df[['age', 'total_spent', 'transaction_count']].corr()
sns.heatmap(correlation, annot=True, ax=axes[1], cmap='coolwarm')
axes[1].set_title('Correlation Matrix')

plt.tight_layout()
plt.show()

print(f"Correlation (Age vs Total Spent): {df['age'].corr(df['total_spent']):.3f}")
```

**4. Feature vs Target Analysis**

```python
# Numeric feature vs Binary target
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Box plot: Target groups
sns.boxplot(x='churned', y='age', data=df, ax=axes[0])
axes[0].set_title('Age by Churn Status')
axes[0].set_ylabel('Age')
axes[0].set_xticklabels(['No Churn', 'Churned'])

# Violin plot: Better visualization of distribution
sns.violinplot(x='churned', y='total_spent', data=df, ax=axes[1])
axes[1].set_title('Total Spent by Churn Status')
axes[1].set_ylabel('Total Spent ($)')

plt.tight_layout()
plt.show()

# Statistical comparison
print("Average age by churn:")
print(df.groupby('churned')['age'].agg(['mean', 'median', 'std']))
```

**5. Correlation with Target**

```python
# Calculate correlation with target variable
target = 'churned'
correlations = df.corr()[target].sort_values(ascending=False)

print("Feature Correlation with Churn:")
print(correlations)

# Visualize
plt.figure(figsize=(10, 6))
correlations[1:].plot(kind='barh')
plt.title('Feature Correlation with Churn')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()
```

**6. Class Imbalance Check**

```python
# Critical for classification problems
print(df['churned'].value_counts())
print(df['churned'].value_counts(normalize=True))

# Visualize
plt.figure(figsize=(8, 4))
df['churned'].value_counts().plot(kind='bar')
plt.title('Churn Distribution')
plt.ylabel('Count')
plt.xticks(rotation=0, labels=['No Churn', 'Churned'])
plt.tight_layout()
plt.show()

# Calculate imbalance ratio
positive_class = (df['churned'] == 1).sum()
negative_class = (df['churned'] == 0).sum()
imbalance_ratio = negative_class / positive_class
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
# If >10:1, need special handling (class weights, resampling)
```

### EDA Questions to Ask

```
1. Data Completeness:
   - How many missing values per column?
   - Are missing patterns random or systematic?

2. Distribution:
   - Is the target well-balanced?
   - Are features normally distributed or skewed?
   - Are there obvious outliers?

3. Relationships:
   - Which features correlate with the target?
   - Are there multicollinearity issues (features highly correlated)?
   - Are relationships linear or non-linear?

4. Time Patterns (if applicable):
   - Are there seasonal patterns?
   - Is there trend over time?
   - Do patterns vary by customer segment?

5. Data Leakage:
   - Are we using future data to predict the past?
   - Are there features that would not be available at prediction time?

6. Segments:
   - Do patterns vary by customer type, geography, time period?
   - Should we build separate models per segment?
```

### Complete EDA Summary Report

```python
def generate_eda_report(df, target_col):
    """
    Generate comprehensive EDA report
    """
    report = f"""
    ========== EDA REPORT ==========
    
    DATASET SHAPE: {df.shape}
    
    MISSING VALUES:
    {df.isnull().sum()}
    
    DATA TYPES:
    {df.dtypes}
    
    TARGET DISTRIBUTION:
    {df[target_col].value_counts()}
    Class Balance: {(df[target_col] == 1).sum() / len(df) * 100:.2f}% positive
    
    NUMERIC FEATURES SUMMARY:
    {df.describe()}
    
    CORRELATION WITH TARGET (Top 10):
    {df.corr()[target_col].nlargest(10)}
    """
    print(report)

generate_eda_report(cleaned_data, 'churned')
```

### Tools Used in EDA

| Tool | Purpose |
|------|---------|
| Pandas | Data exploration, statistics |
| Matplotlib | Basic plotting |
| Seaborn | Statistical visualization |
| Plotly | Interactive visualizations |
| Jupyter | Notebook environment |
| Pandas Profiling | Automated EDA report |
| Apache Spark | Large-scale EDA |

---

## Phase 5: Feature Engineering & Selection

### What is Feature Engineering?

Feature engineering is the process of transforming raw data into meaningful features that help the model learn better patterns.

"Feature engineering is the most important part of machine learning" - Andrew Ng

### Why Feature Engineering Matters

```
Good Features + Simple Model = Often beats Poor Features + Complex Model

Example:
Bad approach: Raw transaction amount → Model accuracy 72%
Good approach: Customer_avg_transaction / customer_total_spent → Accuracy 85%

The ratio (normalized feature) is more informative than raw amount.
```

### Feature Engineering Techniques

**1. Encoding Categorical Variables**

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = cleaned_data.copy()

# Technique 1: Label Encoding (for ordinal categories)
# Use when: Categories have order (Low, Medium, High)
le = LabelEncoder()
df['risk_level_encoded'] = le.fit_transform(df['risk_level'])
# Low=0, Medium=1, High=2

# Technique 2: One-Hot Encoding (for nominal categories)
# Use when: Categories have no order (USA, UK, Canada)
account_type_dummies = pd.get_dummies(df['account_type'], prefix='account_type')
df = pd.concat([df, account_type_dummies], axis=1)
# Creates: account_type_basic, account_type_premium, account_type_vip

# Technique 3: Target Encoding (for high-cardinality)
# Use when: Many categories (100+ cities), need to reduce dimensionality
target_encoding = df.groupby('city')['churned'].mean()
df['city_churn_rate'] = df['city'].map(target_encoding)
# Each city replaced by its actual churn rate

# Technique 4: Frequency Encoding
# Use when: Frequency of category is informative
frequency_encoding = df['country'].value_counts() / len(df)
df['country_frequency'] = df['country'].map(frequency_encoding)
```

**2. Scaling/Normalization Numeric Features**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Technique 1: Standardization (z-score normalization)
# Formula: x_scaled = (x - mean) / std
# Use when: Features are normally distributed, algorithms assume normal dist
scaler = StandardScaler()
df['age_scaled'] = scaler.fit_transform(df[['age']])
# Result: mean=0, std=1

# Technique 2: Min-Max Scaling
# Formula: x_scaled = (x - min) / (max - min)
# Use when: Need features in fixed range [0, 1]
minmax = MinMaxScaler()
df['total_spent_scaled'] = minmax.fit_transform(df[['total_spent']])
# Result: range [0, 1]

# Technique 3: Robust Scaling (handles outliers better)
# Use when: Data has outliers, don't want to remove them
robust = RobustScaler()
df['age_robust'] = robust.fit_transform(df[['age']])
# Uses median and IQR instead of mean and std
```

**3. Creating Time-Based Features**

```python
import pandas as pd

# Extract date features
df['account_creation_date'] = pd.to_datetime(df['account_creation_date'])
df['last_transaction_date'] = pd.to_datetime(df['last_transaction_date'])

# Temporal features
df['account_age_days'] = (pd.Timestamp.now() - df['account_creation_date']).dt.days
df['days_since_last_transaction'] = (pd.Timestamp.now() - df['last_transaction_date']).dt.days

# Cyclic features
df['account_creation_month'] = df['account_creation_date'].dt.month
df['account_creation_quarter'] = df['account_creation_date'].dt.quarter
df['account_creation_dayofweek'] = df['account_creation_date'].dt.dayofweek

# Cyclical encoding (for month: 1-12 should wrap around)
# Use sine/cosine transformation to maintain cyclical nature
df['month_sin'] = np.sin(2 * np.pi * df['account_creation_month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['account_creation_month']/12)
```

**4. Creating Aggregation Features**

```python
# Customer-level aggregations (RFM: Recency, Frequency, Monetary)

# Recency: Days since last purchase
df['recency'] = (df['observation_date'] - df['last_purchase_date']).dt.days

# Frequency: Number of purchases
df['frequency'] = df.groupby('customer_id')['transaction_id'].transform('count')

# Monetary: Total amount spent
df['monetary'] = df.groupby('customer_id')['transaction_amount'].transform('sum')

# Additional aggregations
df['avg_transaction_amount'] = df.groupby('customer_id')['transaction_amount'].transform('mean')
df['std_transaction_amount'] = df.groupby('customer_id')['transaction_amount'].transform('std')
df['max_transaction_amount'] = df.groupby('customer_id')['transaction_amount'].transform('max')
```

**5. Creating Interaction Features**

```python
# Interaction between two features (useful when features interact)

# Ratio features
df['debt_to_income_ratio'] = df['total_debt'] / df['annual_income']
df['avg_transaction_to_total'] = df['avg_transaction_amount'] / df['total_spent']

# Product features
df['age_by_account_age'] = df['age'] * df['account_age_days']

# Polynomial features
df['age_squared'] = df['age'] ** 2
df['total_spent_log'] = np.log1p(df['total_spent'])  # log transformation
```

**6. Feature Selection**

### Why Select Features?

- Reduce dimensionality (fewer features = faster training)
- Remove noise (irrelevant features confuse model)
- Improve interpretability (easier to explain)
- Reduce overfitting

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# Method 1: Statistical Tests (Fast)
# SelectKBest selects k features with highest statistical scores
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("Selected features:", selected_features)

# Method 2: Mutual Information (captures non-linear relationships)
selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Method 3: Feature Importance from Tree Models
# Most practical for real-world use
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)
# Keep top 15 features (example)
top_features = feature_importance.head(15)['feature'].tolist()

# Method 4: Recursive Feature Elimination (Wrapper Method)
# Iteratively removes least important features
from sklearn.feature_selection import RFE
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=15)
X_rfe = rfe.fit_transform(X, y)
selected_rfe = X.columns[rfe.support_]

# Method 5: Correlation-based (Remove multicollinearity)
corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
# Find features with correlation > 0.9
drop_features = [column for column in upper_triangle.columns 
                 if any(upper_triangle[column] > 0.9)]
X_clean = X.drop(columns=drop_features)
```

### Complete Feature Engineering Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_feature_engineering_pipeline():
    """
    Create a preprocessing pipeline
    """
    # Define columns
    numeric_features = ['age', 'total_spent', 'transaction_count', 
                       'account_age_days']
    categorical_features = ['account_type', 'country']
    
    # Numeric transformer: Scale
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer: One-hot encode
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

# Use in model training
from sklearn.linear_model import LogisticRegression

preprocessor = create_feature_engineering_pipeline()

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

full_pipeline.fit(X_train, y_train)
```

### Tools Used in Feature Engineering

| Tool | Purpose |
|------|---------|
| Pandas | Feature creation, manipulation |
| Scikit-learn | Feature selection, scaling |
| Featuretools | Automated feature engineering |
| Category Encoders | Advanced categorical encoding |
| Apache Spark | Large-scale feature engineering |

---

## Phase 6: Model Training, Evaluation & Selection

### What is Model Training?

Training is the process of feeding labeled data to an algorithm so it learns to make predictions.

### Model Selection Strategy

**1. Start Simple (Baseline Models)**

Always start with simple models as baselines. This prevents over-engineering.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Baseline: Logistic Regression
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)
y_proba = baseline_model.predict_proba(X_test)[:, 1]

# Evaluate
print("Baseline Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

**2. Try Multiple Algorithms**

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd

# Dictionary of models to try
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)
}

# Train and evaluate each model
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    })

results_df = pd.DataFrame(results)
print(results_df)
```

### Model Evaluation Metrics

**For Classification Problems:**

```
1. Accuracy: (TP + TN) / Total
   - Overall correctness
   - Misleading with imbalanced data

2. Precision: TP / (TP + FP)
   - Of positive predictions, how many were correct?
   - Important when false positives are costly
   - Churn: Giving incentive to non-churners costs money

3. Recall: TP / (TP + FN)
   - Of actual positives, how many did we catch?
   - Important when false negatives are costly
   - Churn: Missing churners loses customers

4. F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
   - Balanced combination of precision and recall
   - Best when both matter equally

5. ROC-AUC: Area Under Receiver Operating Characteristic Curve
   - Measures model's ability to distinguish classes
   - Ranges 0-1, higher is better
   - Robust to class imbalance

6. PR-AUC: Area Under Precision-Recall Curve
   - Better for imbalanced datasets
   - More informative than ROC-AUC for rare events
```

**Example: Interpreting Metrics for Churn Prediction**

```python
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# [[TN  FP]
#  [FN  TP]]

print(classification_report(y_test, y_pred, 
                          target_names=['No Churn', 'Churned']))

# Visualization
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Business interpretation:
# If Precision=80%: Of 100 customers we predict will churn, 80 actually will
# If Recall=70%: Of all customers who actually churn, we catch 70%
# Cost analysis: 
#   - Cost of retention offer (false positive): $50
#   - Cost of losing customer (false negative): $500
#   - Better to have high recall (catch churners) than high precision
```

**For Regression Problems:**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# MAE: Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")  # Average absolute error in same units as target

# RMSE: Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")  # Penalizes larger errors more

# MAPE: Mean Absolute Percentage Error
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"MAPE: {mape}%")  # Percentage error (good for comparison)

# R-squared: Coefficient of Determination
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2}")  # Proportion of variance explained (0-1, higher better)
```

### Cross-Validation (Robust Evaluation)

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# k-Fold Cross-Validation
# Splits data into k folds, trains k times, evaluates on each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Get scores for each fold
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

print(f"CV Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f}")
print(f"Std: {cv_scores.std():.4f}")

# If mean=0.82 and std=0.03, model is stable and generalizes well
# If mean=0.85 and std=0.12, high variance suggests overfitting risk
```

### Handling Class Imbalance

```python
# Problem: Churn dataset has 85% no-churn, 15% churn
# Simple accuracy becomes misleading (predicting all "no-churn" gives 85%)

# Solution 1: Class Weights
model = LogisticRegression(class_weight='balanced', max_iter=1000)
# Automatically gives more importance to minority class

# Solution 2: SMOTE (Synthetic Minority Oversampling)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# Artificially creates synthetic minority examples

# Solution 3: Threshold Adjustment
# Instead of predicting class if prob > 0.5, use custom threshold
y_pred_custom = (y_proba > 0.3).astype(int)  # Lower threshold catches more churners
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search: Try all combinations
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Random Search: Try random combinations (faster for large spaces)
param_dist = {
    'n_estimators': np.arange(50, 301, 10),
    'max_depth': np.arange(3, 20),
    'min_samples_split': np.arange(2, 11)
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
```

### Model Comparison & Selection

```python
# Compare baseline vs tuned model
baseline_auc = roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:, 1])
tuned_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

print(f"Baseline ROC-AUC: {baseline_auc:.4f}")
print(f"Tuned ROC-AUC: {tuned_auc:.4f}")
print(f"Improvement: {(tuned_auc - baseline_auc):.4f}")

# Select model based on:
# 1. Performance metrics (ROC-AUC, F1, precision/recall trade-off)
# 2. Latency requirements (simpler models are faster)
# 3. Interpretability needs (tree models are more interpretable than neural nets)
# 4. Complexity (don't overfit with overly complex models)
```

### Tools Used in Model Training

| Tool | Purpose |
|------|---------|
| Scikit-learn | ML algorithms, evaluation |
| XGBoost | Gradient boosting (fast, accurate) |
| LightGBM | Faster gradient boosting |
| CatBoost | Handles categorical features well |
| TensorFlow/PyTorch | Deep learning |
| Hyperopt | Bayesian hyperparameter tuning |

---

## Phase 7: Model Deployment

### What is Deployment?

Deployment means taking a trained model and putting it into production so it can make real predictions on new data.

### Deployment Architectures

**1. Batch Prediction (Offline)**

Used when: Predictions needed daily/weekly, not in real-time

```python
# Save trained model
import joblib
joblib.dump(best_model, 'models/churn_model.joblib')

# In production, load and make predictions on new data
def batch_predict(new_data_path):
    """
    Load new data and make predictions
    """
    model = joblib.load('models/churn_model.joblib')
    new_data = pd.read_csv(new_data_path)
    
    # Preprocess
    new_data_processed = preprocessor.transform(new_data)
    
    # Predict
    predictions = model.predict_proba(new_data_processed)[:, 1]
    
    # Save results
    results = pd.DataFrame({
        'customer_id': new_data['customer_id'],
        'churn_probability': predictions,
        'prediction_date': pd.Timestamp.now()
    })
    
    results.to_csv('outputs/churn_predictions.csv', index=False)
    return results

# Schedule with Airflow
# Runs daily at 2 AM, processes all customers, outputs results to database
```

**2. Real-Time API (Online)**

Used when: Predictions needed immediately (fraud detection, recommendations)

```python
# Using FastAPI (modern Python web framework)
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('models/churn_model.joblib')

class PredictionInput(BaseModel):
    customer_id: int
    age: int
    total_spent: float
    transaction_count: int
    account_age_days: int
    account_type: str
    country: str

@app.post("/predict-churn")
async def predict_churn(input_data: PredictionInput):
    """
    Endpoint to predict churn probability
    """
    # Prepare data
    X = pd.DataFrame([input_data.dict()])
    
    # Preprocess
    X_processed = preprocessor.transform(X)
    
    # Predict
    churn_prob = float(model.predict_proba(X_processed)[0, 1])
    prediction = int(model.predict(X_processed)[0])
    
    return {
        'customer_id': input_data.customer_id,
        'churn_probability': churn_prob,
        'prediction': 'Will Churn' if prediction == 1 else 'Will Not Churn',
        'recommendation': 'Send retention offer' if churn_prob > 0.7 else 'No action needed'
    }

# Deploy with Docker
# curl -X POST "http://localhost:8000/predict-churn" \
#      -H "Content-Type: application/json" \
#      -d '{"customer_id": 123, "age": 45, "total_spent": 5000.0, ...}'
```

**3. Containerization with Docker**

Ensures model runs identically everywhere (laptop, server, cloud)

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model and code
COPY churn_model.joblib .
COPY api.py .
COPY preprocessor.joblib .

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t churn-model .
docker run -p 8000:8000 churn-model
```

**4. Model Serving Platforms**

For production-scale serving:

```
Option 1: TensorFlow Serving
- Optimized for deep learning
- High throughput, low latency
- Used by Google internally

Option 2: Seldon
- Kubernetes-native model serving
- Supports any ML framework
- Includes A/B testing, canary deployments

Option 3: KServe
- Built on Kubernetes
- Multi-framework support
- Automatic scaling

Option 4: Cloud Platforms
- AWS SageMaker
- Google Vertex AI
- Azure Machine Learning
- Simple one-click deployment, scales automatically
```

### Deployment Checklist

```
Pre-Deployment:

[✓] Model trained and validated
[✓] Preprocessing pipeline saved with model
[✓] Performance metrics documented
[✓] API endpoints defined
[✓] Input validation implemented
[✓] Error handling added
[✓] Logging configured
[✓] Security reviewed (authentication, rate limiting)

Post-Deployment:

[✓] Monitor prediction latency
[✓] Monitor error rates
[✓] Track model performance (prediction accuracy)
[✓] Monitor data drift (input distribution changes)
[✓] Set up alerting for failures
[✓] Document API usage
[✓] Plan for model updates
```

### Tools Used in Deployment

| Tool | Purpose |
|------|---------|
| Docker | Containerization |
| FastAPI | REST API framework |
| Flask | Lightweight API framework |
| TensorFlow Serving | ML model serving |
| Seldon | Model serving on Kubernetes |
| AWS SageMaker | Cloud model deployment |
| Kubernetes | Container orchestration |

---

## Phase 8: Testing in Production

### Types of Testing

**1. Data Validation**

```python
# Great Expectations: Validate data quality in production
from great_expectations.dataset import PandasDataset

# Define expectations
expectations = PandasDataset(new_data).expect_table_row_count_to_be_between(
    min_value=1000,
    max_value=100000
)

# Check age is in valid range
expectations.expect_column_values_to_be_between(
    column='age',
    min_value=18,
    max_value=120
)

# Check no missing values
expectations.expect_column_values_to_not_be_null(column='customer_id')

# Get validation report
validation_result = expectations.validate()
print(validation_result)
```

**2. Model Performance Monitoring**

```python
# Monitor prediction distribution
# If distribution changes drastically, model might be broken

def monitor_predictions(predictions):
    """
    Track prediction statistics over time
    """
    monitoring_stats = {
        'mean_churn_prob': predictions.mean(),
        'std_churn_prob': predictions.std(),
        'min': predictions.min(),
        'max': predictions.max(),
        'high_risk_count': (predictions > 0.7).sum(),
        'low_risk_count': (predictions < 0.3).sum(),
        'timestamp': pd.Timestamp.now()
    }
    
    # Save to time series database
    # Alert if mean_churn_prob suddenly doubles
    return monitoring_stats
```

**3. A/B Testing (Champion vs Challenger)**

```python
# Split traffic between old model (champion) and new model (challenger)
# Compare performance metrics

def run_ab_test(data, champion_model, challenger_model, split_ratio=0.5):
    """
    Run A/B test between two models
    """
    n = len(data)
    split_point = int(n * split_ratio)
    
    # Champion (old model)
    champion_preds = champion_model.predict(data[:split_point])
    
    # Challenger (new model)
    challenger_preds = challenger_model.predict(data[split_point:])
    
    # Compare metrics
    champion_auc = roc_auc_score(y_true[:split_point], champion_preds)
    challenger_auc = roc_auc_score(y_true[split_point:], challenger_preds)
    
    print(f"Champion AUC: {champion_auc:.4f}")
    print(f"Challenger AUC: {challenger_auc:.4f}")
    
    if challenger_auc > champion_auc + 0.01:  # 1% improvement threshold
        print("Challenger wins! Promote to production.")
        return 'challenger'
    else:
        print("Champion still better. Keep current model.")
        return 'champion'
```

**4. Monitoring Data Drift**

```python
# Data drift: Input distribution changes over time
# Example: Sudden increase in old customers, model was trained on younger customers

from scipy.stats import ks_2samp

def detect_data_drift(X_train, X_new):
    """
    Detect if new data distribution differs from training data
    """
    for column in X_train.columns:
        statistic, p_value = ks_2samp(X_train[column], X_new[column])
        
        if p_value < 0.05:  # Statistically significant difference
            print(f"Data drift detected in column: {column}")
            print(f"p-value: {p_value}")
            # Action: Retrain model on new data
```

### Tools Used in Testing

| Tool | Purpose |
|------|---------|
| Great Expectations | Data validation |
| Evidently | Model monitoring |
| Prometheus | Metrics collection |
| Grafana | Monitoring dashboards |
| MLflow | Model tracking |

---

## Phase 9: Optimization & Continuous Improvement

### Continuous Model Improvement

**1. Retraining Schedule**

```python
# Decide how often to retrain based on:
# - How fast patterns change
# - How much new data accumulates
# - Computational resources

# Example: Churn model retrained weekly
import schedule
import time

def retrain_model():
    """
    Retrain model with latest data
    """
    # Load new data (last 30 days)
    new_data = load_recent_data(days=30)
    
    # Combine with historical data for stability
    combined_data = pd.concat([historical_data, new_data])
    
    # Preprocess, train, validate
    X = preprocessor.fit_transform(combined_data)
    y = combined_data['churned']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train new model
    new_model = RandomForestClassifier(n_estimators=100)
    new_model.fit(X_train, y_train)
    
    # Validate: Is new model better than current model?
    new_auc = roc_auc_score(y_test, new_model.predict_proba(X_test)[:, 1])
    current_auc = evaluate_current_model(X_test, y_test)
    
    if new_auc > current_auc + 0.01:  # Only promote if 1% improvement
        promote_model(new_model)
        print(f"Model updated. New AUC: {new_auc:.4f}")
    else:
        print(f"New model not better. Current AUC: {current_auc:.4f}, New: {new_auc:.4f}")

# Schedule
schedule.every().friday.at("02:00").do(retrain_model)

while True:
    schedule.run_pending()
    time.sleep(60)
```

**2. Feature Importance Analysis**

```python
# Understand which features drive predictions
# Use for model debugging and business insights

def analyze_feature_importance(model, feature_names):
    """
    Analyze which features matter most
    """
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(importance_df)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # Insights: If top 5 features explain 80% of importance,
    # model is interpretable and stable
```

**3. Error Analysis**

```python
# When model makes mistakes, analyze why

def analyze_errors(y_true, y_pred, y_proba, data):
    """
    Analyze model errors for insights
    """
    # False positives: Predicted churn but didn't
    false_positives = (y_pred == 1) & (y_true == 0)
    fp_data = data[false_positives]
    
    print("False Positives (high churn prediction but stayed):")
    print(fp_data[['age', 'total_spent', 'transaction_count']].describe())
    
    # False negatives: Predicted no churn but actually churned
    false_negatives = (y_pred == 0) & (y_true == 1)
    fn_data = data[false_negatives]
    
    print("\nFalse Negatives (missed churners):")
    print(fn_data[['age', 'total_spent', 'transaction_count']].describe())
    
    # Insights:
    # If FNs are old customers with high spend, might need segment-specific models
    # If FPs are new customers, might have cold start problem
```

**4. Model Performance Degradation**

```python
# Monitor if model performance decreases over time

def monitor_performance_degradation(dates, performance_scores):
    """
    Track performance over time
    """
    performance_df = pd.DataFrame({
        'date': dates,
        'auc': performance_scores
    })
    
    # Calculate trend
    performance_df['auc_rolling_avg'] = performance_df['auc'].rolling(7).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(performance_df['date'], performance_df['auc'], label='Daily AUC', alpha=0.5)
    plt.plot(performance_df['date'], performance_df['auc_rolling_avg'], 
             label='7-day Average', linewidth=2)
    plt.axhline(y=0.80, color='r', linestyle='--', label='Minimum Threshold')
    plt.xlabel('Date')
    plt.ylabel('ROC-AUC')
    plt.title('Model Performance Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Alert if AUC drops below threshold for 3 consecutive days
```

### Complete MLDC Workflow Code

```python
class MLDLCPipeline:
    """
    Complete Machine Learning Development Life Cycle
    """
    
    def __init__(self, project_name):
        self.project_name = project_name
        self.preprocessor = None
        self.model = None
        
    def frame_problem(self, business_goal, problem_type, success_metrics):
        """Phase 1: Frame Problem"""
        self.problem_definition = {
            'goal': business_goal,
            'type': problem_type,
            'metrics': success_metrics
        }
        print(f"Problem framed: {business_goal}")
        
    def gather_data(self, query):
        """Phase 2: Gather Data"""
        self.raw_data = pd.read_sql(query, connection)
        print(f"Data gathered: {self.raw_data.shape}")
        
    def process_data(self):
        """Phase 3: Clean & Process"""
        self.cleaned_data = self.raw_data.dropna()
        self.cleaned_data = self.cleaned_data.drop_duplicates()
        print(f"Data cleaned: {self.cleaned_data.shape}")
        
    def eda(self):
        """Phase 4: Exploratory Analysis"""
        print(self.cleaned_data.describe())
        print(f"Churn rate: {self.cleaned_data['churned'].mean():.2%}")
        
    def engineer_features(self):
        """Phase 5: Feature Engineering"""
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        
        numeric_cols = self.cleaned_data.select_dtypes(include='number').columns
        categorical_cols = self.cleaned_data.select_dtypes(include='object').columns
        
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        print("Features engineered")
        
    def train_model(self):
        """Phase 6: Train & Select Model"""
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        X = self.cleaned_data.drop('churned', axis=1)
        y = self.cleaned_data['churned']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X_train_transformed, y_train)
        
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(y_test, self.model.predict_proba(X_test_transformed)[:, 1])
        print(f"Model trained. ROC-AUC: {score:.4f}")
        
        self.X_test = X_test
        self.y_test = y_test
        
    def deploy_model(self):
        """Phase 7: Deploy"""
        import joblib
        joblib.dump(self.model, f'models/{self.project_name}_model.joblib')
        joblib.dump(self.preprocessor, f'models/{self.project_name}_preprocessor.joblib')
        print("Model deployed")
        
    def monitor_model(self):
        """Phase 8 & 9: Monitor & Optimize"""
        print("Monitoring pipeline active")
        print("Scheduled for weekly retraining")

# Usage
pipeline = MLDLCPipeline("churn_prediction")
pipeline.frame_problem(
    business_goal="Reduce churn by 5%",
    problem_type="Classification",
    success_metrics={'precision': 0.80, 'recall': 0.70}
)
pipeline.gather_data("SELECT * FROM customers")
pipeline.process_data()
pipeline.eda()
pipeline.engineer_features()
pipeline.train_model()
pipeline.deploy_model()
pipeline.monitor_model()
```

### Tools Used in Optimization

| Tool | Purpose |
|------|---------|
| MLflow | Experiment tracking, model management |
| Weights & Biases | Experiment tracking, visualization |
| Apache Airflow | Workflow scheduling, orchestration |
| Prefect | Data pipeline orchestration |
| DVC | Data version control |

---

## Summary: MLDC Workflow

| Phase | Goal | Key Activities | Tools |
|-------|------|-----------------|-------|
| 1. Frame Problem | Define clearly | Business goal, metrics, constraints | Docs, SQL |
| 2. Gather Data | Collect raw data | Query databases, APIs, files | SQL, Pandas, APIs |
| 3. Process Data | Clean & prepare | Remove nulls, duplicates, outliers | Pandas, Great Expectations |
| 4. EDA | Understand data | Visualize, analyze distributions | Matplotlib, Seaborn, Pandas |
| 5. Feature Engineering | Create meaningful features | Encode, scale, aggregate, interact | Scikit-learn, Pandas |
| 6. Train & Select | Build & evaluate models | Train multiple models, cross-validate, tune | Scikit-learn, XGBoost |
| 7. Deploy | Put in production | API, containers, serving platform | Docker, FastAPI, Kubernetes |
| 8. Test & Monitor | Validate performance | Data validation, drift detection, A/B test | Great Expectations, Prometheus |
| 9. Optimize | Continuous improvement | Retrain, analyze errors, update features | MLflow, Airflow, Monitoring tools |

---

## Key Takeaways

1. **Problem framing is 50% of success** - A well-defined problem is half-solved

2. **Data quality > Model complexity** - Invest 80% time in data, 20% in algorithms

3. **Always start simple** - Baseline models are fast to build and interpret

4. **Cross-validate properly** - Prevents overfitting and ensures generalization

5. **Monitor in production** - Models degrade over time; retraining is essential

6. **Business metrics matter most** - Accuracy is not the goal; revenue/cost reduction is

7. **Document everything** - Future you will thank present you

8. **Iterate continuously** - ML is not one-time; it's ongoing optimization

---

**ML Development Life Cycle Complete. Ready to build production ML systems!**
