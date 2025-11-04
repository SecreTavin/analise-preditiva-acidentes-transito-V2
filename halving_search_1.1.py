# Código otimizado e comentado para análise preditiva de acidentes de trânsito.
# Utiliza HalvingRandomSearchCV para otimização eficiente de hiperparâmetros.
# Foco em clareza, eficiência e resultados robustos.
# Código final utilizado para trabalho acadêmico.
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*60)
print("ANÁLISE PREDITIVA DE ACIDENTES DE TRÂNSITO")
print("="*60)

# 1. CARREGAMENTO DOS DADOS
engine = create_engine('mysql+mysqlconnector://root:23245623@localhost/analise_transito')
consulta = "SELECT * FROM acidente WHERE gravidade IS NOT NULL"
df = pd.read_sql(consulta, engine)

print(f"\n📊 INFORMAÇÕES DO DATASET:")
print(f"Total de registros: {df.shape[0]:,}")
print(f"Total de colunas: {df.shape[1]}")
print(f"Período dos dados: {df['data_hora'].min()} a {df['data_hora'].max()}")

# 2. EDA
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
gravidade_counts = df['gravidade'].value_counts()
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
plt.pie(gravidade_counts.values, labels=gravidade_counts.index, autopct='%1.1f%%', colors=colors)
plt.title('Distribuição da Gravidade dos Acidentes')
plt.subplot(2, 3, 2)
df['hora'] = pd.to_datetime(df['data_hora']).dt.hour
acidentes_hora = df.groupby('hora').size()
plt.plot(acidentes_hora.index, acidentes_hora.values, marker='o', linewidth=2)
plt.title('Acidentes por Hora do Dia')
plt.xlabel('Hora')
plt.ylabel('Número de Acidentes')
plt.grid(True, alpha=0.3)
plt.subplot(2, 3, 3)
df['dia_semana'] = pd.to_datetime(df['data_hora']).dt.dayofweek
dias = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
acidentes_dia = df.groupby('dia_semana').size()
plt.bar(range(7), acidentes_dia.values, color='skyblue')
plt.title('Acidentes por Dia da Semana')
plt.xlabel('Dia da Semana')
plt.ylabel('Número de Acidentes')
plt.xticks(range(7), dias)
plt.subplot(2, 3, 4)
df['mes'] = pd.to_datetime(df['data_hora']).dt.month
acidentes_mes = df.groupby('mes').size()
plt.bar(acidentes_mes.index, acidentes_mes.values, color='lightgreen')
plt.title('Acidentes por Mês')
plt.xlabel('Mês')
plt.ylabel('Número de Acidentes')
plt.subplot(2, 3, 5)
hora_gravidade = pd.crosstab(df['hora'], df['gravidade'], normalize='index') * 100
sns.heatmap(hora_gravidade, annot=True, fmt='.1f', cmap='YlOrRd')
plt.title('Gravidade por Hora (%)')
plt.ylabel('Hora')
plt.subplot(2, 3, 6)
top_tipos = df['tipo_acidente'].value_counts().head(10)
plt.barh(range(len(top_tipos)), top_tipos.values)
plt.title('Top 10 Tipos de Acidentes')
plt.xlabel('Quantidade')
plt.yticks(range(len(top_tipos)), top_tipos.index)
plt.tight_layout()
plt.savefig('analise_exploratoria_acidentes.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. ENGENHARIA DE FEATURES
print("\n🔧 ENGENHARIA DE FEATURES:")
df['ano'] = pd.to_datetime(df['data_hora']).dt.year
df['mes'] = pd.to_datetime(df['data_hora']).dt.month
df['dia'] = pd.to_datetime(df['data_hora']).dt.day
df['dia_semana'] = pd.to_datetime(df['data_hora']).dt.dayofweek
df['hora'] = pd.to_datetime(df['data_hora']).dt.hour
df['periodo_dia'] = pd.cut(df['hora'], bins=[0,6,12,18,24], labels=['Madrugada','Manhã','Tarde','Noite'], include_lowest=True)
df['fim_semana'] = (df['dia_semana'] >= 5).astype(int)
df['horario_pico'] = ((df['hora'].between(7,9)) | (df['hora'].between(17,19))).astype(int)
y_fatal = (df['gravidade'] == 'Fatal').astype(int)
print(f"Acidentes Fatais: {y_fatal.sum():,} ({y_fatal.mean()*100:.1f}%)")
print(f"Acidentes Não Fatais: {(1-y_fatal).sum():,} ({(1-y_fatal.mean())*100:.1f}%)")

# 4. PREPARAÇÃO DOS DADOS
columns_to_drop = ['id','data_hora','created_at','updated_at','gravidade']
X = df.drop(columns_to_drop, axis=1)
num_cols = X.select_dtypes(include=['number']).columns
cat_cols = X.select_dtypes(include=['object','category']).columns
X[num_cols] = X[num_cols].fillna(0)
for col in cat_cols:
    if X[col].dtype.name == 'category':
        X[col] = X[col].cat.add_categories('Desconhecido')
    X[col] = X[col].fillna('Desconhecido')
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
print(f"Features finais: {X.shape[1]} variáveis")

# 5. MODELAGEM COM RANDOMIZEDSEARCH E AMOSTRA
print("\n🤖 TREINAMENTO DOS MODELOS OTIMIZADOS (RÁPIDO):")
X_train, X_test, y_train, y_test = train_test_split(X, y_fatal, test_size=0.3, random_state=42, stratify=y_fatal)

# Rodar rapidamente com amostra, se desejar:
# X_train = X_train.sample(5000, random_state=42)
# y_train = y_train.loc[X_train.index]

numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

resultados_detalhados = {}

print("\n--- Random Forest (RandomizedSearch) ---")
rf_params = {'n_estimators':[100,200],'max_depth':[10,None],'min_samples_split':[2,5],'min_samples_leaf':[1,2]}
rf_random = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_params, n_iter=6, cv=3, scoring='f1', n_jobs=-1, verbose=2)
rf_random.fit(X_train, y_train)
rf_best = rf_random.best_estimator_
y_pred_rf = rf_best.predict(X_test)
y_proba_rf = rf_best.predict_proba(X_test)[:, 1]
print(f"Melhores parâmetros: {rf_random.best_params_}")
print(f"Acurácia: {accuracy_score(y_test, y_pred_rf):.4f}, AUC-ROC: {roc_auc_score(y_test, y_proba_rf):.4f}")
resultados_detalhados['Random Forest'] = {'accuracy':accuracy_score(y_test,y_pred_rf),'auc':roc_auc_score(y_test,y_proba_rf),'predictions':y_pred_rf,'probabilities':y_proba_rf}

print("\n--- Regressão Logística (RandomizedSearch) ---")
lr_params = {'C':[0.1,1,10],'solver':['liblinear','lbfgs']}
lr_random = RandomizedSearchCV(LogisticRegression(max_iter=1000, random_state=42), lr_params, n_iter=3, cv=3, scoring='f1', verbose=2)
lr_random.fit(X_train, y_train)
lr_best = lr_random.best_estimator_
y_pred_lr = lr_best.predict(X_test)
y_proba_lr = lr_best.predict_proba(X_test)[:, 1]
print(f"Melhores parâmetros: {lr_random.best_params_}")
print(f"Acurácia: {accuracy_score(y_test, y_pred_lr):.4f}, AUC-ROC: {roc_auc_score(y_test, y_proba_lr):.4f}")
resultados_detalhados['Regressão Logística'] = {'accuracy':accuracy_score(y_test,y_pred_lr),'auc':roc_auc_score(y_test,y_proba_lr),'predictions':y_pred_lr,'probabilities':y_proba_lr}

print("\n--- SVM (RandomizedSearch) ---")
svm_params = {'C':[0.1,1],'kernel':['rbf','linear']}
svm_random = RandomizedSearchCV(SVC(probability=True, random_state=42), svm_params, n_iter=2, cv=2, scoring='f1', verbose=2)
svm_random.fit(X_train, y_train)
svm_best = svm_random.best_estimator_
y_pred_svm = svm_best.predict(X_test)
y_proba_svm = svm_best.predict_proba(X_test)[:, 1]
print(f"Melhores parâmetros: {svm_random.best_params_}")
print(f"Acurácia: {accuracy_score(y_test, y_pred_svm):.4f}, AUC-ROC: {roc_auc_score(y_test, y_proba_svm):.4f}")
resultados_detalhados['SVM'] = {'accuracy':accuracy_score(y_test,y_pred_svm),'auc':roc_auc_score(y_test,y_proba_svm),'predictions':y_pred_svm,'probabilities':y_proba_svm}

# 6. ANÁLISE DE RESULTADOS E VISUALIZAÇÕES
print("\n📊 ANÁLISE DETALHADA DOS RESULTADOS:")
resultados_df = pd.DataFrame({
    'Modelo': list(resultados_detalhados.keys()),
    'Acurácia':[r['accuracy'] for r in resultados_detalhados.values()],
    'AUC-ROC':[r['auc'] for r in resultados_detalhados.values()]
})
print(resultados_df.round(4))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes[0,0].bar(resultados_df['Modelo'], resultados_df['Acurácia'], color='skyblue')
axes[0,0].set_title('Acurácia dos Modelos')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,1].bar(resultados_df['Modelo'], resultados_df['AUC-ROC'], color='lightgreen')
axes[0,1].set_title('AUC-ROC dos Modelos')
axes[0,1].tick_params(axis='x', rotation=45)
axes[0,2].plot([0,1],[0,1],'k--',alpha=0.5)
for modelo, dados in resultados_detalhados.items():
    fpr,tpr,_=roc_curve(y_test,dados['probabilities'])
    axes[0,2].plot(fpr,tpr,label=f"{modelo} (AUC={dados['auc']:.3f})")
axes[0,2].set_title('Curvas ROC')
axes[0,2].legend()
cm_rf = confusion_matrix(y_test, resultados_detalhados['Random Forest']['predictions'])
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
axes[1,0].set_title('Matriz de Confusão - RF')
fi = pd.DataFrame({'feature':X.columns,'importance':rf_best.feature_importances_}).sort_values('importance',ascending=True).tail(10)
axes[1,1].barh(fi['feature'],fi['importance'])
axes[1,1].set_title('Top 10 Features - RF')
axes[1,2].hist(resultados_detalhados['Random Forest']['probabilities'][y_test==0],alpha=0.5,label='Não Fatal',bins=50)
axes[1,2].hist(resultados_detalhados['Random Forest']['probabilities'][y_test==1],alpha=0.5,label='Fatal',bins=50)
axes[1,2].set_title('Distribuição Probabilidades - RF')
axes[1,2].legend()
plt.tight_layout()
plt.savefig('resultados_modelos_otimizados.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("RELATÓRIO FINAL PARA ARTIGO CIENTÍFICO")
print("="*60)
melhor = resultados_df.loc[resultados_df['AUC-ROC'].idxmax()]
print(f"\n🏆 MELHOR MODELO: {melhor['Modelo']} (AUC-ROC={melhor['AUC-ROC']:.4f})")
print(f"📈 Taxa de acidentes fatais: {y_fatal.mean()*100:.1f}%")
print(f"⏰ Horário de maior risco: {acidentes_hora.idxmax()}h")
print(f"📅 Dia de maior risco: {dias[acidentes_dia.idxmax()]}")
resultados_df.to_csv('comparacao_modelos_detalhada.csv', index=False)
fi.to_csv('importancia_features.csv', index=False)
print("\n✅ PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
