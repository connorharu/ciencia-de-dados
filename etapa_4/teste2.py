import pandas as pd
from rapidfuzz import process, fuzz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import json
import mysql.connector

# 4.2.1 sem feature engineering da pergunta 2

FILE_PATH = "registros_biodiversidade_ARRUMADO.parquet"
df = pd.read_parquet(FILE_PATH)[["barcode_att","scientificname", "identifiedby_att"]].dropna()

df.head()

def normalize(name):
    return " ".join(name.strip().split())

df["nome_limpo"] = df["scientificname"].apply(normalize)
unique_names = df["nome_limpo"].unique().tolist()

accepted_map = {}
for name in unique_names:
    matches = process.extract(name, unique_names, scorer=fuzz.WRatio, limit=50)
    close = [m[0] for m in matches if m[1] >= 90] # similaridade >=90%
    best = df[df["nome_limpo"].isin(close)]["nome_limpo"].value_counts().idxmax()
    accepted_map[name] = best

df["nome_real"] = df["nome_limpo"].map(accepted_map)
df["correto?"] = (df["nome_limpo"] == df["nome_real"]).astype(int)

X = df[["identifiedby_att"]]
y = df["correto?"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["identifiedby_att"])
])

models = {
    "regressao logistica": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random forest": RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced")
}

results = {}

for name, clf in models.items():
    pipe = Pipeline([
        ("prep", preprocess),
        ("clf", clf)
    ])
    
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:,1]  # probabilidade de correto

    f1 = f1_score(y_test, pred, average="macro")
    print(f"\n{name} - acurácia: {f1:.4f}")
    print(classification_report(y_test, pred))
    
    # salvar probabilidades no dataframe de teste
    df_test = X_test.copy()
    df_test["correto?"] = y_test
    df_test["probabilidade_corretude"] = prob
    df_test["nome_real"] = df.loc[X_test.index, "nome_real"].values
    df_test["scientificname"] = df.loc[X_test.index, "scientificname"].values
    
    # salvar CSV final
    df_test.to_csv(f"scientificname_corrigido_{name.replace(' ','_')}.csv", index=False)
    
    results[name] = f1

for k, v in results.items():
    print(f"{k}: {v:.4f}")

    rf_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced"))
])
rf_pipe.fit(X, y)
prob_all = rf_pipe.predict_proba(X)[:,1]

df_stats = df.copy()
df_stats["probabilidade_corretude"] = prob_all

summary = df_stats.groupby("identifiedby_att")["probabilidade_corretude"].mean().reset_index()
summary = summary.sort_values("probabilidade_corretude")
print("\nmedia de probabilidade de estar correto por identificador:")
print(summary)

summary.to_csv("identifiedby_corretude_sumario.csv", index=False)
print(f"\nesses valores foram salvos em: identifiedby_corretude_sumario.csv")

df_final = df[["scientificname", "identifiedby_att", "correto?", "nome_real"]]
df_final.to_csv("scientificname_corrigido.csv", index=False)
print("\nCSV com nomes atualizados: scientificname_corrigido.csv")

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

db_config = {
    "host": config["db_host"],
    "user": config["db_user"],
    "password": config["db_password"],
    "database": config["db_schema"]
}

conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()
print("conexão estabelecida")

try:
    cursor.execute("""
        ALTER TABLE registros_biodiversidade
        ADD COLUMN scientificname_att TEXT;
    """)
    print("'scientificname_att' criado com sucesso")
except mysql.connector.Error as e:
    if "Duplicate column name" in str(e):
        print("'scientificname_att' já existe")
conn.commit()

try:
    cursor.execute("""
        ALTER TABLE registros_biodiversidade
        ADD COLUMN sinonimo TEXT;
    """)
    print("'sinonimo' criado com sucesso")
except mysql.connector.Error as e:
    if "Duplicate column name" in str(e):
        print("'sinonimo' já existe")
conn.commit()

print(df.columns)

# 4.2.2 feature engineering da pergunta 2

# número total de identificações feitas por cada identificador
ident_counts = df["identifiedby_att"].value_counts().rename("identifications_count")
df["identifications_count"] = df["identifiedby_att"].map(ident_counts)

# Atualizar X para incluir a nova variável
X = df[["identifiedby_att", "identifications_count"]]

# split com a nova variável
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Incluir a coluna numérica
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["identifiedby_att"]),
    ("num", StandardScaler(), ["identifications_count"])
])