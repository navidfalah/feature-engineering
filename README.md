# 🛠️ Feature Engineering Techniques

Welcome to the **Feature Engineering** repository! This project explores various techniques for transforming and enhancing features in machine learning datasets. It covers encoding categorical variables, binning, polynomial features, and automatic feature selection.

---

## 📂 **Project Overview**

This repository demonstrates how to preprocess and engineer features using **Scikit-learn**, **Pandas**, and **mglearn**. It includes:

- **Encoding Categorical Variables**: One-hot encoding, binning, and interaction features.
- **Polynomial Features**: Creating polynomial and interaction terms.
- **Automatic Feature Selection**: Using techniques like SelectPercentile, SelectFromModel, and RFE.
- **Handling Non-Numeric Data**: Transforming timestamps and categorical data into meaningful features.

---

## 🛠️ **Tech Stack**

- **Python**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **mglearn**

---

## 📊 **Datasets**

The project uses the following datasets:
- **Adult Dataset**: For encoding categorical variables and feature transformation.
- **California Housing Dataset**: For polynomial feature engineering.
- **Breast Cancer Dataset**: For automatic feature selection.
- **Citi Bike Dataset**: For time-based feature engineering.

---

## 🧠 **Key Concepts**

### 1. **Encoding Categorical Variables**
- **One-Hot Encoding**: Converts categorical variables into binary vectors.
- **Binning**: Discretizes continuous features into bins.
- **Interaction Features**: Combines features to capture interactions.

### 2. **Polynomial Features**
- Creates polynomial and interaction terms to capture non-linear relationships.

### 3. **Automatic Feature Selection**
- **SelectPercentile**: Selects top features based on statistical tests.
- **SelectFromModel**: Uses model importance scores to select features.
- **RFE (Recursive Feature Elimination)**: Iteratively removes the least important features.

### 4. **Time-Based Feature Engineering**
- Extracts meaningful features from timestamps (e.g., hour, day of the week).

---

## 🚀 **Code Highlights**

### One-Hot Encoding
```python
data_dummies = pd.get_dummies(X)
print("New features:\n", list(data_dummies.columns))
```

### Binning and Encoding
```python
bins = np.linspace(-3, 3, 11)
which_bin = np.digitize(X, bins=bins)
encoder = OneHotEncoder(sparse_output=False)
x_binned = encoder.fit_transform(which_bin)
```

### Polynomial Features
```python
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(X)
print("Polynomial feature names:\n", poly.get_feature_names_out())
```

### Automatic Feature Selection
```python
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
```

### Time-Based Feature Engineering
```python
X_hour = citibike.index.hour.values.reshape(-1, 1)
X_hour_week = np.hstack([X_hour, citibike.index.dayofweek.values.reshape(-1, 1)])
```

---

## 🛠️ **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/navidfalah/feature-engineering.git
   cd feature-engineering
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook feature_engineering.ipynb
   ```

---

## 🤝 **Contributing**

Feel free to contribute to this project! Open an issue or submit a pull request.

---

## 📧 **Contact**

- **Name**: Navid Falah
- **GitHub**: [navidfalah](https://github.com/navidfalah)
- **Email**: navid.falah7@gmail.com
