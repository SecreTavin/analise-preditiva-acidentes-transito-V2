# -----------------------------
# PIPELINE COMPLETO PARA ARTIGO CIENTÍFICO
# Análise e Predição de Acidentes de Trânsito
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.experimental import enable_halving_search_cv  # habilita HalvingRandomSearchCV
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

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

# 2. EDA simplificada
df['hora'] = pd.to_datetime(df['data_hora']).dt.hour
df['dia_semana'] = pd.to_datetime(df['data_hora']).dt.dayofweek
df['mes'] = pd.to_datetime(df['data_hora']).dt.month

plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
plt.pie(df['gravidade'].value_counts().values, labels=df['gravidade'].value_counts().index, autopct='%1.1f%%')
plt.title('Distribuição da Gravidade')
plt.subplot(2,3,2)
ac_h = df.groupby('hora').size()
plt.plot(ac_h.index, ac_h.values, marker='o'); plt.title('Acidentes por Hora')
plt.subplot(2,3,3)
ac_d = df.groupby('dia_semana').size()
plt.bar(range(7), ac_d.values); plt.title('Acidentes por Dia da Semana')
plt.subplot(2,3,4)
ac_m = df.groupby('mes').size()
plt.bar(ac_m.index, ac_m.values); plt.title('Acidentes por Mês')
plt.subplot(2,3,5)
sns.heatmap(pd.crosstab(df['hora'], df['gravidade'], normalize='index')*100, annot=True, fmt='.1f')
plt.title('Gravidade por Hora (%)')
plt.subplot(2,3,6)
top = df['tipo_acidente'].value_counts().head(10)
plt.barh(top.index, top.values); plt.title('Top 10 Tipos')
plt.tight_layout(); plt.savefig('analise_exploratoria_acidentes.png', dpi=300); plt.show()

# 3. ENGENHARIA DE FEATURES
df['periodo_dia'] = pd.cut(df['hora'], bins=[0,6,12,18,24], labels=['Madrugada','Manhã','Tarde','Noite'])
df['fim_semana'] = (df['dia_semana']>=5).astype(int)
df['horario_pico'] = ((df['hora'].between(7,9))|(df['hora'].between(17,19))).astype(int)
y = (df['gravidade']=='Fatal').astype(int)
print(f"\nAcidentes fatais: {y.sum():,} ({y.mean()*100:.1f}%)")

# 4. PREPARAÇÃO
drop_cols = ['id','data_hora','created_at','updated_at','gravidade']
X = df.drop(drop_cols, axis=1)
num = X.select_dtypes('number').columns
cat = X.select_dtypes(['object','category']).columns
X[num] = X[num].fillna(0)
for c in cat:
    if X[c].dtype.name=='category': X[c]=X[c].cat.add_categories('Desconhecido')
    X[c]=X[c].fillna('Desconhecido')
    X[c]=LabelEncoder().fit_transform(X[c].astype(str))
print(f"Features finais: {X.shape[1]} variáveis")

# 5. SPLIT e ESCALA
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
numeric = X_train.select_dtypes(['int64','float64']).columns
sc = StandardScaler()
X_train[numeric]=sc.fit_transform(X_train[numeric])
X_test[numeric]=sc.transform(X_test[numeric])

# 6. OTIMIZAÇÃO COM HALVINGRANDOMSEARCHCV
results = {}
print("\n--- Random Forest (HalvingRandomSearch) ---")
rf_params = {'n_estimators':[100,200,300],'max_depth':[None,10,20],'min_samples_split':[2,5],'min_samples_leaf':[1,2]}
rf_search = HalvingRandomSearchCV(RandomForestClassifier(random_state=42), rf_params, factor=3, cv=3, scoring='f1',
                                  n_jobs=-1, random_state=42, verbose=2)
rf_search.fit(X_train, y_train)
rf = rf_search.best_estimator_
p_rf = rf.predict_proba(X_test)[:,1]
results['Random Forest'] = {'accuracy':accuracy_score(y_test, rf.predict(X_test)),
                            'auc':roc_auc_score(y_test, p_rf),
                            'pred':rf.predict(X_test),
                            'proba':p_rf}

print("\n--- Regressão Logística (HalvingRandomSearch) ---")
lr_params = {'C':[0.01,0.1,1,10],'solver':['liblinear','lbfgs']}
lr_search = HalvingRandomSearchCV(LogisticRegression(max_iter=1000,random_state=42), lr_params,
                                  factor=3, cv=3, scoring='f1', n_jobs=-1, random_state=42, verbose=2)
lr_search.fit(X_train, y_train)
lr = lr_search.best_estimator_
p_lr = lr.predict_proba(X_test)[:,1]
results['Logistic Regression'] = {'accuracy':accuracy_score(y_test, lr.predict(X_test)),
                                  'auc':roc_auc_score(y_test, p_lr),
                                  'pred':lr.predict(X_test),
                                  'proba':p_lr}

print("\n--- SVM (HalvingRandomSearch) ---")
svm_params = {'C':[0.1,1,10],'kernel':['rbf','linear'],'gamma':['scale','auto']}
svm_search = HalvingRandomSearchCV(SVC(probability=True,random_state=42), svm_params,
                                   factor=3, cv=3, scoring='f1', n_jobs=-1, random_state=42, verbose=2)
svm_search.fit(X_train, y_train)
svm = svm_search.best_estimator_
p_svm = svm.predict_proba(X_test)[:,1]
results['SVM'] = {'accuracy':accuracy_score(y_test, svm.predict(X_test)),
                  'auc':roc_auc_score(y_test, p_svm),
                  'pred':svm.predict(X_test),
                  'proba':p_svm}

# 7. RESULTADOS E VISUALIZAÇÕES
df_res = pd.DataFrame({
    'Modelo':list(results.keys()),
    'Acurácia':[v['accuracy'] for v in results.values()],
    'AUC-ROC':[v['auc'] for v in results.values()]
})
print("\nComparação de Modelos:")
print(df_res.round(4))

fig, ax = plt.subplots(1,2,figsize=(12,5))
df_res.plot.bar(x='Modelo',y='Acurácia',ax=ax[0],legend=False); ax[0].set_title('Acurácia')
df_res.plot.bar(x='Modelo',y='AUC-ROC',ax=ax[1],legend=False); ax[1].set_title('AUC-ROC')
plt.tight_layout(); plt.savefig('comparacao_modelos.png',dpi=300); plt.show()

cm = confusion_matrix(y_test, results['Random Forest']['pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues'); plt.title('Matriz de Confusão - RF'); plt.savefig('cm_rf.png',dpi=300); plt.show()

