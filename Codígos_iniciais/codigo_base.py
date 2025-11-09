# -----------------------------
# PIPELINE COMPLETO PARA ARTIGO CIENTÍFICO.
# Análise e Predição de Acidentes de Trânsito.
#ESSE CÓDIGO UTILIZAVA GRIDSEARCH, FOI SUBSTITUÍDO PELO fonte.py, QUE UTILIZA HALVINGRANDOMSEARCHCV.
#NOVO CÓDIGO APRESENTA MELHOR DESEMPENHO E MENOR USO DE CPU E MEMORIA.
#CODIGO DE AMOSTRA PARA GERACÃO DE RELATÓRIO.
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*60)
print("ANÁLISE PREDITIVA DE ACIDENTES DE TRÂNSITO")
print("="*60)

# -----------------------------
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# -----------------------------
engine = create_engine('mysql+mysqlconnector://root:23245623@localhost/analise_transito')
consulta = "SELECT * FROM acidente WHERE gravidade IS NOT NULL"
df = pd.read_sql(consulta, engine)

print(f"\n📊 INFORMAÇÕES DO DATASET:")
print(f"Total de registros: {df.shape[0]:,}")
print(f"Total de colunas: {df.shape[1]}")
print(f"Período dos dados: {df['data_hora'].min()} a {df['data_hora'].max()}")

# -----------------------------
# 2. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# -----------------------------
plt.figure(figsize=(15, 10))

# Distribuição da Gravidade
plt.subplot(2, 3, 1)
gravidade_counts = df['gravidade'].value_counts()
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
plt.pie(gravidade_counts.values, labels=gravidade_counts.index, autopct='%1.1f%%', colors=colors)
plt.title('Distribuição da Gravidade dos Acidentes')

# Acidentes por Hora
plt.subplot(2, 3, 2)
df['hora'] = pd.to_datetime(df['data_hora']).dt.hour
acidentes_hora = df.groupby('hora').size()
plt.plot(acidentes_hora.index, acidentes_hora.values, marker='o', linewidth=2)
plt.title('Acidentes por Hora do Dia')
plt.xlabel('Hora')
plt.ylabel('Número de Acidentes')
plt.grid(True, alpha=0.3)

# Acidentes por Dia da Semana
plt.subplot(2, 3, 3)
df['dia_semana'] = pd.to_datetime(df['data_hora']).dt.dayofweek
dias = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
acidentes_dia = df.groupby('dia_semana').size()
plt.bar(range(7), acidentes_dia.values, color='skyblue')
plt.title('Acidentes por Dia da Semana')
plt.xlabel('Dia da Semana')
plt.ylabel('Número de Acidentes')
plt.xticks(range(7), dias)

# Acidentes por Mês
plt.subplot(2, 3, 4)
df['mes'] = pd.to_datetime(df['data_hora']).dt.month
acidentes_mes = df.groupby('mes').size()
plt.bar(acidentes_mes.index, acidentes_mes.values, color='lightgreen')
plt.title('Acidentes por Mês')
plt.xlabel('Mês')
plt.ylabel('Número de Acidentes')

# Gravidade por Hora (Heatmap)
plt.subplot(2, 3, 5)
hora_gravidade = pd.crosstab(df['hora'], df['gravidade'], normalize='index') * 100
sns.heatmap(hora_gravidade, annot=True, fmt='.1f', cmap='YlOrRd')
plt.title('Gravidade por Hora (%)')
plt.ylabel('Hora')

# Top 10 Tipos de Acidentes
plt.subplot(2, 3, 6)
top_tipos = df['tipo_acidente'].value_counts().head(10)
plt.barh(range(len(top_tipos)), top_tipos.values)
plt.title('Top 10 Tipos de Acidentes')
plt.xlabel('Quantidade')
plt.yticks(range(len(top_tipos)), top_tipos.index)

plt.tight_layout()
plt.savefig('analise_exploratoria_acidentes.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# 3. ENGENHARIA DE FEATURES AVANÇADA
# -----------------------------
print("\n🔧 ENGENHARIA DE FEATURES:")

# Features temporais
df['ano'] = pd.to_datetime(df['data_hora']).dt.year
df['mes'] = pd.to_datetime(df['data_hora']).dt.month
df['dia'] = pd.to_datetime(df['data_hora']).dt.day
df['dia_semana'] = pd.to_datetime(df['data_hora']).dt.dayofweek
df['hora'] = pd.to_datetime(df['data_hora']).dt.hour

# Features categóricas melhoradas
df['periodo_dia'] = pd.cut(df['hora'], 
                           bins=[0, 6, 12, 18, 24], 
                           labels=['Madrugada', 'Manhã', 'Tarde', 'Noite'],
                           include_lowest=True)
df['fim_semana'] = (df['dia_semana'] >= 5).astype(int)
df['horario_pico'] = ((df['hora'].between(7, 9)) | (df['hora'].between(17, 19))).astype(int)

# Variável alvo binária (Fatal vs Não Fatal)
y_fatal = (df['gravidade'] == 'Fatal').astype(int)
print(f"Acidentes Fatais: {y_fatal.sum():,} ({y_fatal.mean()*100:.1f}%)")
print(f"Acidentes Não Fatais: {(1-y_fatal).sum():,} ({(1-y_fatal.mean())*100:.1f}%)")

# -----------------------------
# 4. PREPARAÇÃO DOS DADOS
# -----------------------------
# Remover colunas não necessárias
columns_to_drop = ['id', 'data_hora', 'created_at', 'updated_at', 'gravidade']
X = df.drop(columns_to_drop, axis=1)

# Tratar valores ausentes separadamente
num_cols = X.select_dtypes(include=['number']).columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns

X[num_cols] = X[num_cols].fillna(0)
for col in cat_cols:
    if X[col].dtype.name == 'category':
        X[col] = X[col].cat.add_categories('Desconhecido')
    X[col] = X[col].fillna('Desconhecido')

# Codificar variáveis categóricas
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

print(f"Features finais: {X.shape[1]} variáveis")

# -----------------------------
# 5. MODELAGEM COM OTIMIZAÇÃO DE HIPERPARÂMETROS
# -----------------------------
print("\n🤖 TREINAMENTO DOS MODELOS OTIMIZADOS:")

# Split dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y_fatal, test_size=0.3, random_state=42, stratify=y_fatal
)

# Identificar colunas numéricas após encoding
numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns

# Normalização apenas das numéricas para SVM e MLP
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

resultados_detalhados = {}

# Random Forest otimizado
print("\n--- Random Forest (GridSearch) ---")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)
y_proba_rf = rf_best.predict_proba(X_test)[:, 1]
print(f"Melhores parâmetros: {rf_grid.best_params_}")
print(f"Acurácia: {accuracy_score(y_test, y_pred_rf):.4f}, AUC-ROC: {roc_auc_score(y_test, y_proba_rf):.4f}")
resultados_detalhados['Random Forest'] = {'accuracy':accuracy_score(y_test,y_pred_rf),'auc':roc_auc_score(y_test,y_proba_rf),'predictions':y_pred_rf,'probabilities':y_proba_rf}

# Regressão Logística otimizada
print("\n--- Regressão Logística (GridSearch) ---")
lr_params = {'C': [0.1,1,10,100],'solver':['liblinear','lbfgs']}
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), lr_params, cv=5, scoring='f1')
lr_grid.fit(X_train, y_train)
lr_best = lr_grid.best_estimator_
y_pred_lr = lr_best.predict(X_test)
y_proba_lr = lr_best.predict_proba(X_test)[:, 1]
print(f"Melhores parâmetros: {lr_grid.best_params_}")
print(f"Acurácia: {accuracy_score(y_test, y_pred_lr):.4f}, AUC-ROC: {roc_auc_score(y_test, y_proba_lr):.4f}")
resultados_detalhados['Regressão Logística'] = {'accuracy':accuracy_score(y_test,y_pred_lr),'auc':roc_auc_score(y_test,y_proba_lr),'predictions':y_pred_lr,'probabilities':y_proba_lr}

# SVM otimizado
print("\n--- SVM (GridSearch) ---")
svm_params = {'C':[0.1,1,10],'kernel':['rbf','linear'],'gamma':['scale','auto']}
svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_params, cv=3, scoring='f1')
svm_grid.fit(X_train, y_train)
svm_best = svm_grid.best_estimator_
y_pred_svm = svm_best.predict(X_test)
y_proba_svm = svm_best.predict_proba(X_test)[:, 1]
print(f"Melhores parâmetros: {svm_grid.best_params_}")
print(f"Acurácia: {accuracy_score(y_test, y_pred_svm):.4f}, AUC-ROC: {roc_auc_score(y_test, y_proba_svm):.4f}")
resultados_detalhados['SVM'] = {'accuracy':accuracy_score(y_test,y_pred_svm),'auc':roc_auc_score(y_test,y_proba_svm),'predictions':y_pred_svm,'probabilities':y_proba_svm}

# -----------------------------
# 6. ANÁLISE DE RESULTADOS E VISUALIZAÇÕES
# -----------------------------
print("\n📊 ANÁLISE DETALHADA DOS RESULTADOS:")
resultados_df = pd.DataFrame({
    'Modelo': list(resultados_detalhados.keys()),
    'Acurácia':[r['accuracy'] for r in resultados_detalhados.values()],
    'AUC-ROC':[r['auc'] for r in resultados_detalhados.values()]
})
print(resultados_df.round(4))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Acurácia
axes[0,0].bar(resultados_df['Modelo'], resultados_df['Acurácia'], color='skyblue')
axes[0,0].set_title('Acurácia dos Modelos')
axes[0,0].tick_params(axis='x', rotation=45)

# AUC-ROC
axes[0,1].bar(resultados_df['Modelo'], resultados_df['AUC-ROC'], color='lightgreen')
axes[0,1].set_title('AUC-ROC dos Modelos')
axes[0,1].tick_params(axis='x', rotation=45)

# Curvas ROC
axes[0,2].plot([0,1],[0,1],'k--',alpha=0.5)
for modelo, dados in resultados_detalhados.items():
    fpr,tpr,_=roc_curve(y_test,dados['probabilities'])
    axes[0,2].plot(fpr,tpr,label=f"{modelo} (AUC={dados['auc']:.3f})")
axes[0,2].set_title('Curvas ROC')
axes[0,2].legend()

# Matriz Confusão RF
cm_rf = confusion_matrix(y_test, resultados_detalhados['Random Forest']['predictions'])
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
axes[1,0].set_title('Matriz de Confusão - RF')

# Importância features RF
fi = pd.DataFrame({'feature':X.columns,'importance':rf_best.feature_importances_}).sort_values('importance',ascending=True).tail(10)
axes[1,1].barh(fi['feature'],fi['importance'])
axes[1,1].set_title('Top 10 Features - RF')

# Distribuição probabilidades RF
axes[1,2].hist(resultados_detalhados['Random Forest']['probabilities'][y_test==0],alpha=0.5,label='Não Fatal',bins=50)
axes[1,2].hist(resultados_detalhados['Random Forest']['probabilities'][y_test==1],alpha=0.5,label='Fatal',bins=50)
axes[1,2].set_title('Distribuição Probabilidades - RF')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('resultados_modelos_otimizados.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# 7. RELATÓRIO FINAL PARA O ARTIGO
# -----------------------------
print("\n" + "="*60)
print("RELATÓRIO FINAL PARA ARTIGO CIENTÍFICO")
print("="*60)

melhor = resultados_df.loc[resultados_df['AUC-ROC'].idxmax()]
print(f"\n🏆 MELHOR MODELO: {melhor['Modelo']} (AUC-ROC={melhor['AUC-ROC']:.4f})")
print(f"📈 Taxa de acidentes fatais: {y_fatal.mean()*100:.1f}%")
print(f"⏰ Horário de maior risco: {acidentes_hora.idxmax()}h")
print(f"📅 Dia de maior risco: {dias[acidentes_dia.idxmax()]}")

# Salvar resultados
resultados_df.to_csv('comparacao_modelos_detalhada.csv', index=False)
fi.to_csv('importancia_features.csv', index=False)
print("\n✅ PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
