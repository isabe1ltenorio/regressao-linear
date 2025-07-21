import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns

app = Flask(__name__)

# Carregar e preparar os dados com novas features
def load_and_preprocess_data():
    df = pd.read_csv('df_merged.csv')
    df = df.drop(columns=['Data', 'Gols'])  # Remover colunas redundantes
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Criar novas variáveis derivadas
    df['Eficiencia_Finalizacao'] = df['Chutes_No_Gol'] / df['Total_Chutes']
    df['Intensidade_Ataque'] = df['Escanteios'] * df['Impedimentos']
    
    # Tratar divisão por zero
    df['Eficiencia_Finalizacao'] = df['Eficiencia_Finalizacao'].replace([np.inf, -np.inf], np.nan)
    
    # Remover linhas com valores faltantes críticos
    df = df.dropna(subset=['Gols_Pro', 'Gols_Contra'])
    return df

df = load_and_preprocess_data()

# Combinações inválidas
invalid_combinations = {
    'Gols_Pro': ['Gols_Contra', 'Tiros_Meta', 'Cartoes_Vermelhos'],
    'Gols_Contra': ['Gols_Pro', 'Penaltis_batidos', 'Impedimentos']
}

# Variáveis recomendadas para cada alvo
feature_recommendations = {
    'Gols_Pro': [
        ['Chutes_No_Gol', 'Eficiencia_Finalizacao', 'Intensidade_Ataque'],
        ['Chutes_No_Gol', 'Escanteios', 'Penaltis_batidos'],
        ['Intensidade_Ataque', 'Faltas_Provocadas', 'Posse']
    ],
    'Gols_Contra': [
        ['Tiros_Meta', 'Faltas_Cometidas', 'Cartoes_Amarelos'],
        ['Escanteios', 'Laterais', 'Cartoes_Vermelhos'],
        ['Faltas_Cometidas', 'Cartoes_Amarelos', 'Posse']
    ]
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Obter seleções do usuário
        target = request.form.get('target')
        features = request.form.getlist('features')
        
        # Validar seleções
        if not target or not features:
            return render_template('index.html', 
                                   columns=df.columns,
                                   recommendations=feature_recommendations,
                                   error="Selecione pelo menos uma variável dependente e uma independente")
        
        # Verificar combinações inválidas
        invalid_vars = []
        if target in invalid_combinations:
            for feature in features:
                if feature in invalid_combinations[target]:
                    invalid_vars.append(feature)
        
        if invalid_vars:
            error_msg = f"Combinação inválida: {target} não pode ser prevista por {', '.join(invalid_vars)}"
            return render_template('index.html', 
                                   columns=df.columns,
                                   recommendations=feature_recommendations,
                                   error=error_msg)
        
        # Preparar dados
        df_clean = df.dropna(subset=[target] + features)
        
        # Verificar se há dados suficientes
        if len(df_clean) < 5:
            return render_template('index.html', 
                                   columns=df.columns,
                                   recommendations=feature_recommendations,
                                   error=f"Dados insuficientes após limpeza (apenas {len(df_clean)} registros). Selecione outras variáveis.")
        
        X = df_clean[features].values
        y = df_clean[target].values
        
        # Padronizar os dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Treinar modelo
        model = LinearRegression()
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Gerar gráficos
        plot_urls = generate_plots(y, y_pred, X_scaled, features, model)
        
        # Coeficientes para interpretação
        coefficients = pd.Series(model.coef_, index=features).sort_values(ascending=False)
        
        return render_template('results.html', 
                              mae=mae,
                              r2=r2,
                              plot_urls=plot_urls,
                              target=target,
                              features=features,
                              coefficients=coefficients.to_dict())
    
    return render_template('index.html', 
                          columns=df.columns,
                          recommendations=feature_recommendations)

def generate_plots(y, y_pred, X, feature_names, model):
    plots = {}
    
    # Gráfico de valores reais vs previstos
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=y_pred)
    plt.plot([min(y), max(y)], [min(y), max(y)], '--r')
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Previstos')
    plt.title('Valores Reais vs Previstos')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['real_vs_pred'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Gráfico de resíduos
    plt.figure(figsize=(10, 6))
    residuals = y - y_pred
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Resíduos')
    plt.title('Análise de Resíduos')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['residuals'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Distribuição de resíduos
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Resíduos')
    plt.title('Distribuição dos Resíduos')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['residuals_dist'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Coeficientes do modelo
    if len(feature_names) > 0:
        plt.figure(figsize=(10, 6))
        coefs = pd.Series(model.coef_, index=feature_names)
        coefs = coefs.sort_values()
        coefs.plot(kind='barh')  # Barras horizontais para melhor leitura
        plt.title('Coeficientes do Modelo')
        plt.xlabel('Magnitude')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plots['coefficients'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    
    return plots

if __name__ == '__main__':
    app.run(debug=True)