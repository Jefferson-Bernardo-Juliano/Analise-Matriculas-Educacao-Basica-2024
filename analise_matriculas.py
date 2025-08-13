# # EDA Template — Projeto: Evasão Escolar (Ensino Médio Público)

# !pip install pandas matplotlib scikit-learn missingno openpyxl requests tabulate scipy
print('Ambiente pronto. Verifique que você tem pandas, matplotlib, scikit-learn, requests, scipy.')

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

plt.rcParams['figure.figsize'] = (10,5)
plt.rcParams['figure.dpi'] = 100

print('Imports carregados')

# %%
DATA_PATHS = {
    "censo": "../data/censo_escolar_2024.csv",
    "pnad_uf_grupo_idade": "../data/pnad_uf_grupo_idade.csv",
    "pnad_regiao_grupo_idade": "../data/pnad_regiao_grupo_idade.csv"
}


# %%
def small_preview(df, n=5):
    print(df.head(n))
    print('Shape:', df.shape)
    print(df.head(n))
    print('\nMissing per column:')
    print(df.isna().sum().sort_values(ascending=False).head(20))


def save_clean(df, name='cleaned.csv'):
    os.makedirs('output', exist_ok=True)
    path = os.path.join('output', name)
    df.to_csv(path, index=False)
    print('Saved to', path)

print('Helper functions definidas')

# %%
datasets = {}

# carregar os datasets
for key, path in DATA_PATHS.items():
    try:
        datasets[key] = pd.read_csv(path, sep=';', low_memory=False, encoding='latin1')
        print(f"Carregado: {key} -> {path}")
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {path}. Faça o download da fonte correspondente e coloque em data/ ou atualize DATA_PATHS.")

# Preview (opcional)
for k, df in datasets.items():
    print('\n### Preview:', k)
    small_preview(df)

# %%
# Limpeza e pré-processamento - Checklist

# Exemplo genérico para 'censo'
if 'censo' in datasets:
    df = datasets['censo']
    # remover duplicatas por identificador único da matrícula
    if 'NU_INSCRICAO' in df.columns:
        df = df.drop_duplicates(subset=['NU_INSCRICAO'])
    else:
        df = df.drop_duplicates()

    # Normalizar nomes de colunas (minúsculas)
    df.columns = [c.lower().strip() for c in df.columns]

    # Tratar datas
    for col in df.columns:
        if 'data' in col or 'ano' in col:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass

    # criar presença percentual se houver faltas e dias letivos
    if set(['faltas', 'dias_letivos']).issubset(df.columns):
        df['presenca_pct'] = 1 - (df['faltas'] / df['dias_letivos'])

    datasets['censo_clean'] = df
    print('Limpeza básica do censo concluída. Salvando...')
    save_clean(df, 'censo_clean.csv')

else:
    print('Censo não encontrado nos datasets carregados.')

# %%
# Análise descritiva

if 'censo_clean' in datasets:
    df = datasets['censo_clean']
    # Estatísticas descritivas
    print(df.describe(include='all'))

    # Histograma de uma variável numérica
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 0:
        col = num_cols[0]
        plt.figure()
        plt.hist(df[col].dropna(), bins=30)
        plt.title(f'Distribuição de {col}')
        plt.xlabel(col)
        plt.ylabel('Frequência')
        plt.show()

    # Boxplot rápido
    if len(num_cols) > 1:
        col2 = num_cols[1]
        plt.figure()
        plt.boxplot([df[col].dropna(), df[col2].dropna()], labels=[col, col2])
        plt.title('Boxplot - exemplo')
        plt.show()
else:
    print('Rode a célula de carregamento de dados primeiro.')

# %%
# Correlação (Pearson) — apenas variáveis numéricas
if 'censo_clean' in datasets:
    df = datasets['censo_clean']
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] > 1:
        corr = num.corr(method='pearson')
        plt.figure(figsize=(10,8))
        plt.imshow(corr, aspect='auto')
        plt.colorbar()
        plt.title('Matriz de Correlação (visualização)')
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.show()
    else:
        print('Poucas variáveis numéricas para correlação')
else:
    print('Censo limpo não disponível para correlação')

# %%
# Teste qui-quadrado — associação entre duas categóricas
from scipy.stats import chi2_contingency

if 'censo_clean' in datasets:
    df = datasets['censo_clean']

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(cat_cols) >= 2:
        ctab = pd.crosstab(df[cat_cols[0]], df[cat_cols[1]])
        stat, p, dof, expected = chi2_contingency(ctab.fillna(0))
        print('Chi2:', stat, 'p-value:', p)
    else:
        print('Poucas colunas categóricas detectadas para teste qui-quadrado')
else:
    print('Censo limpo não disponível para chi-square')

# %%
# Quick model (RandomForest) para obter importância de features — DEMO
# Requer uma coluna target binária chamada 'evasao' (0/1) no dataset
if 'censo_clean' in datasets:
    df = datasets['censo_clean'].copy()
    if 'evasao' in df.columns:
        # Exemplo simples: selecionar variáveis numéricas
        X = df.select_dtypes(include=[np.number]).drop(columns=['evasao'], errors='ignore').fillna(0)
        y = df['evasao'].astype(int)
        if X.shape[0] > 50 and len(X.columns) > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(classification_report(y_test, y_pred))
            print('ROC AUC:', roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
            feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
            print('\nTop 20 features:\n', feat_imp.head(20))
        else:
            print('Dataset muito pequeno ou sem variáveis numéricas para treinar o RF de exemplo')
    else:
        print("Coluna 'evasao' não encontrada no censo limpo. Crie/defina o target para rodar o modelo.")
else:
    print('Censo limpo não disponível para modelagem rápida')

# %%
# Outputs finais — salvar dataset tratado e variáveis importantes (exemplo)
if 'censo_clean' in datasets:
    save_clean(datasets['censo_clean'], 'censo_final_for_model.csv')
print('Fim do template EDA')

