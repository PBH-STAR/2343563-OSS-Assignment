# ================================================
# OSS개발 개인과제 - Breast Cancer 분류 모델
# 데이터셋: load_breast_cancer (sklearn)
# 모델: DecisionTree → RandomForest (성능 비교)
# ================================================

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier   
from sklearn.metrics import accuracy_score, classification_report 

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
print(classification_report(y_test, dt_pred, target_names=data.target_names))

# ── 4. 모델 학습 (Random Forest - 성능 향상) ─────── 
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("\n=== [모델 2] Random Forest (n_estimators=100) 결과 ===")
print(f"정확도(Accuracy): {rf_acc:.4f}  ({rf_acc*100:.2f}%)")
print(classification_report(y_test, rf_pred, target_names=data.target_names))

# ── 5. 성능 비교 요약 ─────────────────────────────
print("\n=== 모델 성능 비교 ===")
print(f"{'모델':<30} {'정확도':>10}")
print("-" * 42)
print(f"{'Decision Tree (baseline)':<30} {dt_acc*100:>9.2f}%")
print(f"{'Random Forest (n=100)':<30} {rf_acc*100:>9.2f}%")
print(f"{'향상폭':<30} {(rf_acc - dt_acc)*100:>+9.2f}%p")
