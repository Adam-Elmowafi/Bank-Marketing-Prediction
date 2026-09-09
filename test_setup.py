"""
Test Script for Bank Deposit Predictor

Run this script to verify:
1. All dependencies are installed
2. bank.csv is readable
3. Data preprocessing works correctly
"""

import sys
import pandas as pd

print("=" * 60)
print("🧪 Bank Deposit Predictor - Setup Verification")
print("=" * 60)

# Test 1: Check Python version
print("\n1️⃣ Checking Python version...")
version = sys.version_info
if version.major >= 3 and version.minor >= 8:
    print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
else:
    print(f"   ❌ Python {version.major}.{version.minor} (needs 3.8+)")
    sys.exit(1)

# Test 2: Check required packages
print("\n2️⃣ Checking required packages...")
required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'plotly': 'plotly',
    'streamlit': 'streamlit',
}

all_installed = True
for module_name, package_name in required_packages.items():
    try:
        __import__(module_name)
        print(f"   ✅ {package_name}")
    except ImportError:
        print(f"   ❌ {package_name} (run: pip install {package_name})")
        all_installed = False

if not all_installed:
    print("\n❌ Some packages are missing!")
    print("   Fix: pip install -r requirements.txt")
    sys.exit(1)

# Test 3: Check bank.csv exists and load it
print("\n3️⃣ Checking bank.csv...")
try:
    df = pd.read_csv('bank.csv', sep=',')
    print(f"   ✅ File loaded successfully")
    print(f"   ✅ Shape: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print(f"   ❌ bank.csv not found!")
    print(f"   📍 Place bank.csv in the same folder as app.py")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ Error loading CSV: {e}")
    sys.exit(1)

# Test 4: Check required columns
print("\n4️⃣ Checking required columns...")
required_cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 
                 'housing', 'loan', 'contact', 'day', 'month', 'campaign', 
                 'pdays', 'previous', 'poutcome']
target_col = 'deposit' if 'deposit' in df.columns else 'y'

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"   ⚠️  Missing columns: {missing_cols}")
else:
    print(f"   ✅ All required input features found")

if target_col not in df.columns:
    print(f"   ❌ Target column '{target_col}' not found!")
    sys.exit(1)
else:
    print(f"   ✅ Target column '{target_col}' found")

# Test 5: Check data quality
print("\n5️⃣ Checking data quality...")
print(f"   • Duplicates: {df.duplicated().sum()}")
print(f"   • Missing values: {df.isnull().sum().sum()}")
print(f"   • Target value counts:")
for val, count in df[target_col].value_counts().items():
    pct = count / len(df) * 100
    print(f"     - {val}: {count} ({pct:.1f}%)")

# Test 6: Test preprocessing
print("\n6️⃣ Testing data preprocessing...")
try:
    # Simulate preprocessing
    df_copy = df.copy()
    
    # Convert target
    if df_copy[target_col].dtype == 'object':
        df_copy[target_col] = df_copy[target_col].astype(str).str.strip().str.lower().map({'no': 0, 'yes': 1})
    
    # Separate X and y
    X = df_copy.drop(columns=[target_col])
    y = df_copy[target_col]
    
    # Binary encoding
    binary_cols = ['default', 'housing', 'loan']
    for col in binary_cols:
        if col in X.columns:
            X[col] = X[col].astype(str).str.strip().str.lower().replace({'yes': 1, 'no': 0}).fillna(0).astype(int)
    
    # One-hot encoding
    categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    existing_cats = [c for c in categorical_cols if c in X.columns]
    X_encoded = pd.get_dummies(X, columns=existing_cats, drop_first=True, dtype=int)
    
    print(f"   ✅ Preprocessing successful")
    print(f"   ✅ Features after encoding: {X_encoded.shape[1]}")
    print(f"   ✅ Training samples: {X_encoded.shape[0]}")
    
except Exception as e:
    print(f"   ❌ Preprocessing error: {e}")
    sys.exit(1)

# Test 7: Test model training
print("\n7️⃣ Testing model training...")
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_acc = lr.score(X_test_scaled, y_test)
    
    # Train SVM
    svm = SVC(random_state=42, probability=True)
    svm.fit(X_train_scaled, y_train)
    svm_acc = svm.score(X_test_scaled, y_test)
    
    print(f"   ✅ Logistic Regression trained (accuracy: {lr_acc:.2%})")
    print(f"   ✅ SVM trained (accuracy: {svm_acc:.2%})")
    
except Exception as e:
    print(f"   ❌ Model training error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed!
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED! Your setup is ready!")
print("=" * 60)
print("\n🚀 Next step: Run the app with:")
print("   streamlit run app.py")
print("\n" + "=" * 60)
