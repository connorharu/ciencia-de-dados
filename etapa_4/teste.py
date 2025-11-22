import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_parquet("registros_biodiversidade_ARRUMADO.parquet")

# Manter só colunas relevantes
df = df[["basisofrecord", "yearcollected", "monthcollected", "daycollected"]]

# Remover linhas q basisofrecord é nulo
df = df.dropna(subset=["basisofrecord"])


# pré processamento

label_basis = LabelEncoder()
df["basis_encoded"] = label_basis.fit_transform(df["basisofrecord"])

X = df[["yearcollected", "monthcollected", "daycollected"]]
y = df["basis_encoded"]

imputer = SimpleImputer(strategy="most_frequent")
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.25, random_state=42, stratify=y
)


# 4.2.1 regressão logística pra pergunta 1

model_log = LogisticRegression(max_iter=500)
model_log.fit(X_train, y_train)

y_pred_log = model_log.predict(X_test)

print("Resultados da regressão logística:")
print("Acurácia:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# 4.2.1 random forest pra pergunta 1

model_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

print("Resultados do random forest:")
print("Acurácia:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Importância das variáveis
importances = pd.Series(model_rf.feature_importances_, 
index=["yearcollected", "monthcollected", "daycollected"])
print("\nImportância das variáveis:")
print(importances)

### 4.2.2 Feature Engineering -----------------------------------------------------------------------

df = pd.read_parquet("registros_biodiversidade_ARRUMADO.parquet")

# Manter só colunas relevantes da pergunta 1
df = df[["yearcollected", "monthcollected", "daycollected", "basisofrecord"]]

# Remover linhas sem valores essenciais
df = df.dropna(subset=["yearcollected", "basisofrecord"])

# Criar "collection_age"
df["collection_age"] = 2025 - df["yearcollected"]

# Criar flag coleta moderna
df["is_modern"] = np.where(df["yearcollected"] >= 2000, 1, 0)

# Transformar mês em estação
def get_season(month):
    if month in [12, 1, 2]:
        return "summer"
    if month in [3, 4, 5]:
        return "autumn"
    if month in [6, 7, 8]:
        return "winter"
    if month in [9, 10, 11]:
        return "spring"
    return "unknown"

df["season"] = df["monthcollected"].apply(get_season)

# Remover anos invalidos

df = df[(df["yearcollected"] >= 1) & (df["yearcollected"] <= 2025)]

# Converter ano/mês/dia para inteiros
df["yearcollected"] = df["yearcollected"].astype(int)
df["monthcollected"] = df["monthcollected"].fillna(1).astype(int).clip(1, 12)
df["daycollected"] = df["daycollected"].fillna(1).astype(int).clip(1, 31)

# Calcular o dayofyear sem erro
df["dayofyear"] = df.apply(
    lambda row: pd.Timestamp(
        row["yearcollected"],
        row["monthcollected"],
        row["daycollected"]
    ).dayofyear,
    axis=1
)

# Encoding da variável alvo
le = LabelEncoder()
df["basis_encoded"] = le.fit_transform(df["basisofrecord"])

# One-hot das categorias criadas
df = pd.get_dummies(df, columns=["season"], drop_first=True)

# Seleção de varíaveis e modelo complexo

X = df.drop(columns=["basisofrecord", "basis_encoded"])
y = df["basis_encoded"]

# Normalização
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4.2.2 random forest com feature engineering pra pergunta 1
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("\n==============================")
print("Random forest com feature engineering")
print("==============================")
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Importância das variáveis
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nImportancia das varíaveis:")
print(importances)

# 4.2.2 regressão logística com feature engineering pra pergunta 1

log_reg = LogisticRegression(
    max_iter=500,
    multi_class="auto",
    solver="lbfgs"
)

log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)

print("Regressão logística com feature engineering")
print("Acurácia:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
