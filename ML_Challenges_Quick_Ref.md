# ML Challenges & Problems - Quick Reference Guide

## One-Page Summary for Every Challenge

---

## 1. DATA COLLECTION

**Problem:** Getting data is hard (expensive, legal issues, scraping blocked)

**Web Scraping Issues:**
- Legal: ToS violations, GDPR/CCPA compliance
- Technical: CAPTCHA, IP blocking, JavaScript rendering
- Ethical: Copyright, rate limiting

**API Limitations:**
- Rate limits: Only X requests per hour
- Quota limits: Only Y requests per month
- Unstable: APIs change format or disappear

**Solutions:**
```python
# Use APIs responsibly
import time
response = requests.get(url, headers={'Authorization': f'Bearer {token}'})
time.sleep(2)  # Respectful delay

# Use Selenium for dynamic content
driver = webdriver.Chrome()
driver.get(url)
element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "data")))

# Better: Use free datasets
from sklearn.datasets import load_iris
from tensorflow import keras
keras.datasets.mnist.load_data()
```

**Cost:** $100 - $50,000+ depending on method

---

## 2. INSUFFICIENT / LABELED DATA

**Problem:** Need lots of labeled data, but expensive/time-consuming to label

**Costs:**
- Manual labeling: $0.10 - $10+ per sample
- 100,000 samples: $10,000 - $500,000
- Medical imaging: Even more expensive

**Solutions:**

```python
# 1. Data Augmentation: Make more samples from existing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(rotation_range=20, horizontal_flip=True)

# 2. Transfer Learning: Use pre-trained models
from tensorflow.keras.applications import MobileNetV2
model = MobileNetV2(weights='imagenet')
# Only retrain top layers with your data

# 3. Semi-Supervised: Use unlabeled data
from sklearn.semi_supervised import LabelSpreading
model = LabelSpreading()  # Spreads labels to unlabeled data

# 4. Weak Supervision: Use noisy rules
def weak_label(text):
    if 'great' in text: return 1
    elif 'bad' in text: return 0
    else: return -1  # Uncertain

# 5. Active Learning: Label only most informative samples
# Start with small labeled set
# Ask human to label 10 most uncertain samples
# Retrain, repeat

# 6. Crowdsourcing: Many cheap annotators
# 100 people label 100 samples each = 10K labels cheap
```

**Bottom line:** Labeling is bottleneck, use these strategies!

---

## 3. NON-REPRESENTATIVE DATA

**Problem:** Dataset doesn't match real-world population

**Two types:**

**Sampling Noise (Random):**
- Natural variation when sampling
- Solution: Increase sample size (reduces by √n)

**Sampling Bias (Systematic):**
- Survey in library only: Misses non-studious students
- Man-on-street interview: Only healthy, mobile people
- Email dataset from Gmail only: Misses other providers

```python
# Detect bias
population_stats = {'gender': {'M': 0.49, 'F': 0.51}}
dataset_stats = {'gender': {'M': 0.70, 'F': 0.30}}
# Divergence: 0.20 (20%) → BIASED!

# Solutions:
# 1. Stratified sampling: Preserve proportions
from sklearn.model_selection import StratifiedShuffleSplit

# 2. Reweight samples: Give less weight to overrepresented groups
weights = np.array([0.49/0.70 if gender=='M' else 0.51/0.30 for gender in data])
model.fit(X, y, sample_weight=weights)

# 3. Collect representative data: Ensure all groups included
```

**Impact:** Unfair models, poor generalization

---

## 4. POOR DATA QUALITY

**Definition:** Garbage IN → Garbage OUT

**MIT: 82% of ML projects stall due to DATA QUALITY issues**

**Types:**
- Missing values: Incomplete data
- Outliers: Extreme values skewing results
- Inconsistencies: "john" vs "JOHN" vs "john "
- Duplicates: Same record twice
- Errors: Wrong data

```python
# Solutions:
import pandas as pd
import numpy as np

# Missing values
data.fillna(data.mean())  # Simple
from sklearn.impute import KNNImputer
imputer.fit_transform(data)  # Better

# Outliers
from sklearn.ensemble import IsolationForest
iso = IsolationForest()
outliers = iso.fit_predict(data)

# Inconsistencies
data['name'] = data['name'].str.strip().str.lower()
data['gender'] = data['gender'].map({'M': 'male', 'F': 'female'})

# Duplicates
data.drop_duplicates()

# Validation
def validate_data(data):
    print("Missing:", data.isnull().sum())
    print("Duplicates:", data.duplicated().sum())
    print("Summary:", data.describe())
```

**Impact:** Model accuracy drops 10-50%+

---

## 5. IRRELEVANT FEATURES

**Problem:** Including garbage features hurts performance

```
More irrelevant features = Worse model!
```

**Solutions:**

```python
# 1. Filter Methods (fast, statistical)
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 2. Wrapper Methods (uses model)
from sklearn.feature_selection import RFE
rfe = RFE(estimator, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

# 3. Embedded Methods (built into model)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, y)
importances = rf.feature_importances_

# 4. Domain Expertise
# "Color doesn't matter for house prices"
# Keep: square_feet, bedrooms, location
# Remove: owner_height, owner_age
```

**Impact:** 10-50% performance improvement possible!

---

## 6. OVERFITTING

**Problem:** Model learns noise, not patterns

```
Training Accuracy: 99%
Test Accuracy: 40%  ← Model overfitted!
```

**Causes:**
- Model too complex
- Too little training data
- No regularization
- Too much training time

**Solutions:**

```python
# 1. Regularization (L1/L2)
from sklearn.linear_model import Ridge, Lasso
ridge = Ridge(alpha=1.0)  # Higher alpha = more regularization

# 2. Reduce complexity
model = RandomForestClassifier(
    max_depth=5,           # Shallow trees
    min_samples_split=10   # More samples needed to split
)

# 3. More training data
# Get more samples, reduce noise

# 4. Early stopping (Neural Networks)
early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X, y, callbacks=[early_stop])

# 5. Dropout (Neural Networks)
layers.Dropout(0.3)  # Drop 30% of neurons

# 6. Cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
if huge gap between train/test: Overfitting!
```

**Detection:**
- Gap between train and test accuracy > 10%
- Train accuracy near 100%, test accuracy much lower

---

## 7. UNDERFITTING

**Problem:** Model too simple to learn patterns

```
Training Accuracy: 60%
Test Accuracy: 58%  ← Model underfitted!
```

**Causes:**
- Model too simple
- Insufficient training
- Poor features
- Too much regularization

**Solutions:**

```python
# 1. More complex model
simple = LinearRegression()        # Underfitting risk
complex = RandomForestClassifier() # Better

# 2. Feature engineering
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(X)

# 3. Train longer
model.fit(X, y, epochs=100)  # More iterations

# 4. Reduce regularization
ridge = Ridge(alpha=0.01)  # Lower alpha = less regularization

# 5. Better features
# Domain expertise: "Add weather data for crop prediction"
```

**Detection:**
- Low accuracy on both train and test
- No improvement with more data

---

## 8. SOFTWARE INTEGRATION

**Problem:** Model works offline, fails in production

**Challenges:**
- Environment mismatch (different Python version, packages)
- Model serving (how to use in production)
- Feature inconsistency (computed differently)
- Monitoring (can't debug failures)

```python
# Solutions:

# 1. Containerization (Docker)
# Same environment everywhere!
dockerfile = """
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model.pkl .
CMD ["python", "api.py"]
"""

# 2. Flask API
from flask import Flask, request
app = Flask(__name__)
model = pickle.load(open('model.pkl'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    return {'prediction': float(model.predict([data['features']])[0])}

# 3. Feature Store (consistent features)
from feast import FeatureStore
feature_store = FeatureStore(repo_path='.')
# Same features in training and production!

# 4. Logging
import logging
logging.info(f"Input: {X}")
logging.info(f"Prediction: {pred}")
logging.error(f"Error: {e}")

# 5. Monitoring
from prometheus_client import Counter
prediction_counter = Counter('predictions_total', 'Total')
prediction_counter.inc()
```

**Cost:** $50K - $500K for infrastructure

---

## 9. OFFLINE LEARNING & CONCEPT DRIFT

**Problem:** Model trained on old data performs poorly on new data

```
Trained on: 2020 data (normal market)
Deployed in: 2024 (market changed!)
Result: Model predictions way off
```

**Examples:**
- Spam filter trained 2020, new spam types 2024
- Stock predictor trained pre-pandemic, deployed 2021 crash
- Product recommendation learned old preferences, users changed

**Solutions:**

```python
# 1. Detect drift
from scipy.stats import ks_2samp
stat, p_value = ks_2samp(X_train, X_recent)
if p_value < 0.05:
    print("⚠️  Data distribution changed!")

# 2. Online learning (continuous updates)
from sklearn.linear_model import SGDClassifier
model = SGDClassifier()
for batch in new_data:
    model.partial_fit(batch)

# 3. Scheduled retraining
# Retrain model every week with recent data
schedule.every().monday.at("02:00").do(retrain)

# 4. Ensemble of models
# Combine old model + recent model
# Recent gets more weight
```

**Cost:** Requires continuous monitoring and retraining

---

## 10. COST

**Problem:** ML projects are expensive!

**Breakdown:**
- Data labeling: $10K - $1M
- Computing (GPUs): $1K - $100K/month
- Personnel (DS, Engineer): $150K - $250K/year each
- Infrastructure: $5K - $100K
- **Total Year 1: $200K - $2M+**

```python
def estimate_cost(dataset_size, complexity):
    annotation_cost = dataset_size * {'simple': 0.10, 'complex': 5.00}[complexity]
    personnel = {'simple': 1, 'complex': 4}[complexity] * 75_000
    compute = {'simple': 5_000, 'complex': 100_000}[complexity]
    total = annotation_cost + personnel + compute
    return total

# Example: 50K samples, moderate complexity
cost = estimate_cost(50_000, 'complex')
# → ~$375K
```

**Solutions:**

```python
# 1. Transfer Learning
# Use pre-trained model: Save $400K!
from transformers import AutoModel
model = AutoModel.from_pretrained('bert-base-uncased')
model.fit(your_data, epochs=5)

# 2. Data Augmentation
# 10K→100K samples: Save $40K
from imgaug import augmenters
aug = augmenters.SomeOf([...])
augmented = [aug(img) for img in images for _ in range(10)]

# 3. AutoML
# Replace $150K data scientist with $50/month tool
from h2o import automl
aml = automl.H2OAutoML(max_models=20)
aml.train(X=X, y=y)

# 4. Smart prioritization
# Only build if ROI > 300%
roi = (annual_benefit - annual_cost) / annual_cost
```

**When NOT to build:**
- ROI negative
- Data doesn't exist
- Simple rules work better
- Ethical concerns

---

## DECISION MATRIX: Which Challenge Do You Have?

| Symptom | Challenge | Fix |
|---------|-----------|-----|
| Can't get data | Data Collection | Use APIs, crowdsource, buy |
| Data expensive to label | Insufficient Data | Transfer learning, data augmentation |
| Model biased, unfair | Non-Representative | Stratified sampling, reweight |
| Garbage values, missing | Poor Quality | Cleaning, validation |
| Performance unchanged after adding features | Irrelevant Features | Feature selection |
| Train: 99%, Test: 40% | Overfitting | Regularization, more data |
| Train: 60%, Test: 58% | Underfitting | Complex model, features |
| Works offline, fails live | Integration | Docker, monitoring, logging |
| Accuracy drops over time | Offline Learning | Online updates, retraining |
| Too expensive | Cost | Transfer learning, AutoML |

---

## QUICK CHECKLIST

Before starting ML project:
- [ ] Understand ROI (is it worth it?)
- [ ] Have/can get data (quality & quantity)
- [ ] Budget allocated ($200K minimum)
- [ ] Team available (data scientist, engineer)
- [ ] Ethical/regulatory OK
- [ ] Success metric defined

During development:
- [ ] Clean data thoroughly
- [ ] Check for biases (representative?)
- [ ] Monitor train vs test accuracy
- [ ] Feature selection done
- [ ] Cross-validation used
- [ ] Documented everything

Before deployment:
- [ ] Containerized (Docker)
- [ ] Monitoring/logging ready
- [ ] Rollback plan
- [ ] A/B testing plan
- [ ] Retraining schedule

After deployment:
- [ ] Monitor performance daily
- [ ] Detect data drift
- [ ] Retrain on schedule
- [ ] Log everything
- [ ] Handle edge cases

---

**Print this for your desk!** Reference when facing real ML challenges.
