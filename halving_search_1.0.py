# -----------------------------
# PIPELINE COMPLETO PARA ARTIGO CIENTÍFICO
# Análise e Predição de Acidentes de Trânsito
# Primeira versão utilizando HalvingRandomSearchCV.
# código de amostra para geração de relatório.
# -----------------------------
from sklearn.experimental import enable_halving_search_cv  # habilita HalvingRandomSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*60)
print("ANÁLISE PREDITIVA DE ACIDENTES DE TRÂNSITO")
print("="*60)

# 1. CARREGAMENTO DOS DADOS
engine = create_engine('mysql+mysqlconnector://root:23245623@localhost/analise_transito')
df = pd.read_sql("SELECT * FROM acidente WHERE gravidade IS NOT NULL", engine)

print(f"\n📊 INFORMAÇÕES DO DATASET:")
print(f"Total de registros: {df.shape[0]:,}")
print(f"Total de colunas: {df.shape[1]}")
print(f"Período: {df['data_hora'].min()} a {df['data_hora'].max()}")

# 2. EDA resumida
df['hora'] = pd.to_datetime(df['data_hora']).dt.hour
df['dia_semana'] = pd.to_datetime(df['data_hora']).dt.dayofweek
df['mes'] = pd.to_datetime(df['data_hora']).dt.month

# 3. ENGENHARIA DE FEATURES
df['periodo_dia'] = pd.cut(df['hora'], bins=[0,6,12,18,24],
    labels=['Madrugada','Manhã','Tarde','Noite'], include_lowest=True)
df['fim_semana'] = (df['dia_semana'] >= 5).astype(int)
df['horario_pico'] = ((df['hora'].between(7,9))|(df['hora'].between(17,19))).astype(int)

# Alvo
y_fatal = (df['gravidade']=='Fatal').astype(int)

# 4. PREPARAÇÃO DOS DADOS
drop_cols = ['id','data_hora','created_at','updated_at','gravidade']
X = df.drop(drop_cols, axis=1)
num_cols = X.select_dtypes('number').columns
cat_cols = X.select_dtypes(['object','category']).columns
X[num_cols] = X[num_cols].fillna(0)
for c in cat_cols:
    if X[c].dtype.name=='category':
        X[c]=X[c].cat.add_categories('Desconhecido')
    X[c]=X[c].fillna('Desconhecido')
    X[c]=LabelEncoder().fit_transform(X[c].astype(str))

# 5. SPLIT E ESCALA
X_train,X_test,y_train,y_test = train_test_split(
    X,y_fatal,test_size=0.3,random_state=42,stratify=y_fatal)
numeric = X_train.select_dtypes(['int64','float64']).columns
scaler = StandardScaler()
X_train[numeric]=scaler.fit_transform(X_train[numeric])
X_test[numeric]=scaler.transform(X_test[numeric])

# 6. OTIMIZAÇÃO COM HALVINGRANDOMSEARCHCV
resultados = {}

# Random Forest
print("\n--- RF (HalvingRandomSearch) ---")
rf_params = {
    'n_estimators':[100,200,300],
    'max_depth':[None,10,20],
    'min_samples_split':[2,5],
    'min_samples_leaf':[1,2]
}
rf_halving = HalvingRandomSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params, factor=3, cv=3, scoring='f1',
    n_jobs=-1, random_state=42, verbose=2
)
rf_halving.fit(X_train, y_train)
rf = rf_halving.best_estimator_
p_rf = rf.predict_proba(X_test)[:,1]
resultados['RF'] = {
    'accuracy':accuracy_score(y_test, rf.predict(X_test)),
    'auc':roc_auc_score(y_test, p_rf),
    'pred':rf.predict(X_test),
    'proba':p_rf
}

# Regressão Logística
print("\n--- LR (HalvingRandomSearch) ---")
lr_params = {'C':[0.01,0.1,1,10], 'solver':['liblinear','lbfgs']}
lr_halving = HalvingRandomSearchCV(
    LogisticRegression(max_iter=1000,random_state=42),
    lr_params, factor=3, cv=3, scoring='f1',
    n_jobs=-1, random_state=42, verbose=2
)
lr_halving.fit(X_train, y_train)
lr = lr_halving.best_estimator_
p_lr = lr.predict_proba(X_test)[:,1]
resultados['LR'] = {
    'accuracy':accuracy_score(y_test, lr.predict(X_test)),
    'auc':roc_auc_score(y_test, p_lr),
    'pred':lr.predict(X_test),
    'proba':p_lr
}

# SVM
print("\n--- SVM (HalvingRandomSearch) ---")
svm_params = {'C':[0.1,1,10],'kernel':['rbf','linear'],'gamma':['scale','auto']}
svm_halving = HalvingRandomSearchCV(
    SVC(probability=True,random_state=42),
    svm_params, factor=3, cv=3, scoring='f1',
    n_jobs=-1, random_state=42, verbose=2
)
svm_halving.fit(X_train, y_train)
svm = svm_halving.best_estimator_
p_svm = svm.predict_proba(X_test)[:,1]
resultados['SVM'] = {
    'accuracy':accuracy_score(y_test, svm.predict(X_test)),
    'auc':roc_auc_score(y_test, p_svm),
    'pred':svm.predict(X_test),
    'proba':p_svm
}

# 7. RESULTADOS RESUMIDOS
df_res = pd.DataFrame({
    'Modelo':list(resultados.keys()),
    'Acurácia':[r['accuracy'] for r in resultados.values()],
    'AUC-ROC':[r['auc'] for r in resultados.values()]
})
print("\nComparação:")
print(df_res.round(4))

# 8. PRINCIPAIS MUDANÇAS
# • Substituição de GridSearchCV por HalvingRandomSearchCV
#   reduzindo drasticamente tempo e memória.
# • Uso de factor=3 elimina progressivamente configurações ruins.
# • Mantém uso de todo o dataset em cada rodada.
# • Verbose=2 para monitorar progresso no terminal.
