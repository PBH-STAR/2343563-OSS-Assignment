# ================================================
# OSS개발 개인과제 - Breast Cancer 분류 모델
# 데이터셋: load_breast_cancer (sklearn)
# ================================================

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier        
from sklearn.metrics import accuracy_score              

# ── 1. 데이터셋 로드 ──────────────────────────────
data = load_breast_cancer()
X = data.data
y = data.target

print("=== 데이터셋 정보 ===")
print(f"전체 샘플 수: {X.shape[0]}")
print(f"특성(feature) 수: {X.shape[1]}")
print(f"클래스: {data.target_names}")

# ── 2. Train / Test Split ─────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n학습 데이터: {X_train.shape[0]}개")
print(f"테스트 데이터: {X_test.shape[0]}개")

# ── 3. 모델 학습 (Decision Tree - 베이스라인) ────── 
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

print("\n=== [모델 1] Decision Tree 결과 ===")
print(f"정확도(Accuracy): {dt_acc:.4f}  ({dt_acc*100:.2f}%)")
