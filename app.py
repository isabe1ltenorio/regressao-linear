import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns
import os

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'


datasets = {}


PRETTY_NAMES = {
    'Gols_Pro': 'Gols Pró',
    'Gols_Contra': 'Gols Contra',
    'Total_Chutes': 'Total de Chutes',
    'Chutes_No_Gol': 'Chutes no Gol',
    'Chutes_Falta': 'Chutes de Falta',
    'Penaltis_batidos': 'Pênaltis Convertidos',
    'Penaltis tentados': 'Pênaltis Tentados',
    'Laterais': 'Arremessos Laterais',
    'Escanteios': 'Escanteios',
    'Tiros_Meta': 'Tiros de Meta',
    'Cartoes_Amarelos': 'Cartões Amarelos',
    'Cartoes_Vermelhos': 'Cartões Vermelhos',
    'Faltas_Cometidas': 'Faltas Cometidas',
    'Faltas_Provocadas': 'Faltas Sofridas',
    'Impedimentos': 'Impedimentos',
    'Posse': 'Posse de Bola (%)',
    'Eficiencia_Finalizacao': 'Eficiência de Finalização',
    'Intensidade_Ataque': 'Intensidade de Ataque'
}


TEAM_LOGOS = {
    'Barcelona': 'barcelona.png',
    'Atlético de Madrid': 'atletico.png'
}

def load_and_preprocess_data(file_path, team_name):
    df = pd.read_csv(file_path)
    # Remover colunas indesejadas
    df = df.drop(columns=['Data', 'Gols'], errors='ignore')
    df = df.apply(pd.to_numeric, errors='coerce')
    

    if 'Total_Chutes' in df.columns and 'Chutes_No_Gol' in df.columns:
        df['Eficiencia_Finalizacao'] = df['Chutes_No_Gol'] / df['Total_Chutes']
        df['Eficiencia_Finalizacao'] = df['Eficiencia_Finalizacao'].replace([np.inf, -np.inf], np.nan)
    
    if 'Escanteios' in df.columns and 'Impedimentos' in df.columns:
        df['Intensidade_Ataque'] = df['Escanteios'] * df['Impedimentos']
    

    df['Time'] = team_name
    
    return df


try:
    datasets['Barcelona'] = load_and_preprocess_data('barcelona.csv', 'Barcelona')
    print(f"Dataset Barcelona carregado com {datasets['Barcelona'].shape[0]} registros")
except Exception as e:
    print(f"Erro ao carregar dataset do Barcelona: {e}")

try:
    datasets['Atlético de Madrid'] = load_and_preprocess_data('atletico.csv', 'Atlético de Madrid')
    print(f"Dataset Atlético de Madrid carregado com {datasets['Atlético de Madrid'].shape[0]} registros")
except Exception as e:
    print(f"Erro ao carregar dataset do Atlético de Madrid: {e}")


invalid_combinations = {
    'Gols_Pro': ['Gols_Contra', 'Tiros_Meta', 'Cartoes_Vermelhos'],
    'Gols_Contra': ['Gols_Pro', 'Penaltis_batidos', 'Impedimentos']
}

feature_recommendations = {
    'Gols_Pro': [
        ['Chutes_No_Gol', 'Escanteios', 'Penaltis_batidos'],
        ['Intensidade_Ataque', 'Faltas_Provocadas', 'Posse']
    ],
    'Gols_Contra': [
        ['Tiros_Meta', 'Faltas_Cometidas', 'Cartoes_Amarelos'],
        ['Faltas_Cometidas', 'Cartoes_Amarelos', 'Cartoes_Vermelhos']
    ]
}

@app.route('/')
def index():
    if not datasets:
        return render_template('team_selection.html', 
                               teams=[],
                               datasets={},
                               team_logos=TEAM_LOGOS,
                               error="Nenhum dataset carregado.")
    
    return render_template('team_selection.html', 
                           teams=list(datasets.keys()),
                           datasets=datasets,
                           team_logos=TEAM_LOGOS)


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():

    if request.method == 'POST':
        team = request.form.get('team')
    else: 
        team = request.args.get('team')
    
    if not team or team not in datasets:
        return redirect(url_for('index'))
    
    df = datasets[team]
    columns = [col for col in df.columns if col not in ['Time']]
    
    return render_template('index.html', 
                          columns=columns,
                          recommendations=feature_recommendations,
                          team=team,
                          pretty_names=PRETTY_NAMES,
                          team_logos=TEAM_LOGOS)

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    team = request.form.get('team')
    if team not in datasets:
        return redirect(url_for('index'))
    
    df = datasets[team]
    

    target = request.form.get('target')
    features = request.form.getlist('features')
    
    # Validar seleções
    if not target or not features:
        columns = [col for col in df.columns if col not in ['Time']]
        return render_template('index.html', 
                              columns=columns,
                              recommendations=feature_recommendations,
                              team=team,
                              pretty_names=PRETTY_NAMES,
                              error="Selecione pelo menos uma variável dependente e uma independente")
    

    invalid_vars = []
    if target in invalid_combinations:
        for feature in features:
            if feature in invalid_combinations[target]:
                invalid_vars.append(feature)
    
    if invalid_vars:
        error_msg = f"Combinação inválida: {PRETTY_NAMES.get(target, target)} não pode ser prevista por {', '.join([PRETTY_NAMES.get(var, var) for var in invalid_vars])}"
        columns = [col for col in df.columns if col not in ['Time']]
        return render_template('index.html', 
                              columns=columns,
                              recommendations=feature_recommendations,
                              team=team,
                              pretty_names=PRETTY_NAMES,
                              error=error_msg)
    

    df_clean = df.dropna(subset=[target] + features)
    
    if len(df_clean) < 5:
        columns = [col for col in df.columns if col not in ['Time']]
        return render_template('index.html', 
                              columns=columns,
                              recommendations=feature_recommendations,
                              team=team,
                              pretty_names=PRETTY_NAMES,
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
    plot_urls = generate_plots(y, y_pred, X_scaled, features, model, team, target)
    
    # Coeficientes para interpretação
    coefficients = pd.Series(model.coef_, index=features).sort_values(ascending=False)
    
    return render_template('results.html', 
                          mae=mae,
                          r2=r2,
                          plot_urls=plot_urls,
                          target=target,
                          features=features,
                          coefficients=coefficients.to_dict(),
                          team=team,
                          pretty_names=PRETTY_NAMES,
                          team_logos=TEAM_LOGOS)

def generate_plots(y, y_pred, X, feature_names, model, team, target):
    plots = {}
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=y_pred)
    plt.plot([min(y), max(y)], [min(y), max(y)], '--r')
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Previstos')
    plt.title(f'Valores Reais vs Previstos - {team}')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['real_vs_pred'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Análise de resíduos
    plt.figure(figsize=(10, 6))
    residuals = y - y_pred
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Resíduos')
    plt.title(f'Análise de Resíduos - {team}')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plots['residuals'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # Distribuição de resíduos
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Resíduos')
    plt.title(f'Distribuição dos Resíduos - {team}')
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
        coefs.plot(kind='barh')
        plt.title(f'Coeficientes do Modelo - {team}')
        plt.xlabel('Magnitude')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plots['coefficients'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    
    return plots

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)