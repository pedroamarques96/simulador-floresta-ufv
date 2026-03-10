import os
import numpy as np
import pandas as pd 
import matplotlib
matplotlib.use('Agg') # OBRIGATÓRIO PARA APLICAÇÕES WEB (Evita travamento do servidor)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from scipy.optimize import minimize_scalar

# Configuração visual
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 150

# ==============================================================================
# 1. PARÂMETROS DO CENÁRIO E ESPAÇAMENTO (Valores Padrão - Alteráveis pelo App)
# ==============================================================================
IDADE_INICIAL = 24
HORIZONTE = 60
AGENDA = {
    24: 0.10,  # 10% de mortalidade aos 24 meses
} 

DIST_LINHA = 3.0       
DIST_ENTRELINHA = 2.0  
DIST_DIAG = np.sqrt(DIST_LINHA**2 + DIST_ENTRELINHA**2)

LINHAS = 33
COLUNAS = 33

# ==============================================================================
# 2. PARÂMETROS DAS EQUAÇÕES (BIOMETRIA)
# ==============================================================================
GAMMA = 8.88215118266297
BETA = 11.3684667510947

B0_VTCC = -9.96279390282725
B1_VTCC = 2.12845873614494
B2_VTCC = 0.787242303602132

B0_HT = -0.355914700751589
B1_HT = 1.28266887955174

B1_DAP = -9.53359653018803
B2_DAP = -0.812691121907373

B1_HT_PROJ = -10.3329143998905
B2_HT_PROJ = -0.626933278635436

# ==============================================================================
# 3. PARÂMETROS DA MORTALIDADE (LOGÍSTICA)
# ==============================================================================
B0_MORT = -0.8852   
B1_MORT = -6.1832   

def obter_vizinhos_8(linha, coluna, max_lin, max_col):
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    vizinhos = []
    for dl, dc in offsets:
        nl, nc = linha + dl, coluna + dc
        if 0 <= nl < max_lin and 0 <= nc < max_col:
            vizinhos.append((nl, nc))
    return vizinhos

def calcular_idrp_matricial(dap_matriz, dist_linha, dist_entre, dist_diag):
    rows, cols = dap_matriz.shape
    ab_matriz = (np.pi * (dap_matriz**2)) / 40000
    ab_clean = np.nan_to_num(ab_matriz, nan=0.0)
    
    w_linha = 1.0 / dist_linha
    w_entre = 1.0 / dist_entre
    w_diag  = 1.0 / dist_diag
    
    soma_pond = np.zeros((rows, cols))
    soma_pesos = np.zeros((rows, cols))
    
    def shift(arr, dy, dx):
        res = np.zeros_like(arr)
        src_y_a, src_y_b = max(0, -dy), min(rows, rows - dy)
        src_x_a, src_x_b = max(0, -dx), min(cols, cols - dx)
        dst_y_a, dst_y_b = max(0, dy), min(rows, rows + dy)
        dst_x_a, dst_x_b = max(0, dx), min(cols, cols + dx)
        res[dst_y_a:dst_y_b, dst_x_a:dst_x_b] = arr[src_y_a:src_y_b, src_x_a:src_x_b]
        return res

    vizinhos_config = [
        (-1, 0, w_linha), (1, 0, w_linha),
        (0, -1, w_entre), (0, 1, w_entre),
        (-1, -1, w_diag), (-1, 1, w_diag),
        (1, -1, w_diag), (1, 1, w_diag)
    ]
    
    for dy, dx, w in vizinhos_config:
        ab_viz = shift(ab_clean, dy, dx) 
        mask_viz = ab_viz > 0            
        soma_pond += np.where(mask_viz, ab_viz * w, 0)
        soma_pesos += np.where(mask_viz, w, 0)
        
    media_viz = np.divide(soma_pond, soma_pesos, out=np.zeros_like(soma_pond), where=soma_pesos!=0)
    media_geral_ab = np.nanmean(ab_matriz)
    epsilon = media_geral_ab * 0.01 if media_geral_ab > 0 else 0.0001
    idrp = ab_matriz / (media_viz + epsilon)
    return idrp

def gerar_floresta_completa(linhas, colunas, beta, gamma, seed):
    if seed is not None:
        np.random.seed(seed)
        
    n_total = linhas * colunas
    daps = beta * np.random.weibull(a=gamma, size=n_total)
    daps = np.round(daps, 2)
    np.random.shuffle(daps)
    matriz_dap = daps.reshape((linhas, colunas))
    
    matriz_ht = np.exp(B0_HT + B1_HT * np.log(matriz_dap))
    matriz_ht = np.round(matriz_ht, 2)
    
    matriz_vol = np.exp(B0_VTCC + B1_VTCC * np.log(matriz_dap) + B2_VTCC * np.log(matriz_ht))
    matriz_vol = np.round(matriz_vol, 5)
    
    matriz_idrp = calcular_idrp_matricial(matriz_dap, DIST_LINHA, DIST_ENTRELINHA, DIST_DIAG)
    matriz_idrp = np.round(matriz_idrp, 4)
    
    return {
        "DAP": matriz_dap, "HT": matriz_ht,
        "VOL": matriz_vol, "IID": matriz_idrp 
    }

def calcular_ile_dinamico(death_date_matrix, idade_atual, dist_linha, dist_entre, dist_diag):
    rows, cols = death_date_matrix.shape
    diff_meses = idade_atual - death_date_matrix
    tmort_anos = (diff_meses / 12.0) + 1.0
    
    fator_inv_tempo = np.zeros_like(death_date_matrix)
    mask_mortas = ~np.isnan(death_date_matrix)
    fator_inv_tempo[mask_mortas] = 1.0 / tmort_anos[mask_mortas]
    
    w_linha = 1.0 / dist_linha
    w_entre = 1.0 / dist_entre
    w_diag  = 1.0 / dist_diag
    
    ile_dinamico = np.zeros((rows, cols))
    
    def shift(arr, dy, dx):
        res = np.zeros_like(arr)
        src_y_a, src_y_b = max(0, -dy), min(rows, rows - dy)
        src_x_a, src_x_b = max(0, -dx), min(cols, cols - dx)
        dst_y_a, dst_y_b = max(0, dy), min(rows, rows + dy)
        dst_x_a, dst_x_b = max(0, dx), min(cols, cols + dx)
        res[dst_y_a:dst_y_b, dst_x_a:dst_x_b] = arr[src_y_a:src_y_b, src_x_a:src_x_b]
        return res

    vizinhos_config = [
        (-1, 0, w_linha), (1, 0, w_linha),  
        (0, -1, w_entre), (0, 1, w_entre),  
        (-1, -1, w_diag), (-1, 1, w_diag),  
        (1, -1, w_diag), (1, 1, w_diag)
    ]
    
    for dy, dx, w in vizinhos_config:
        ile_dinamico += shift(fator_inv_tempo, dy, dx) * w
        
    return ile_dinamico

def simular_floresta_compensatoria(floresta_dict, idade_inicial, horizonte_meses, cronograma_mortalidade, beta_b3=0.0):
    dap_atual = floresta_dict['DAP'].copy()
    ht_atual = floresta_dict['HT'].copy()
    
    idrp_atual = calcular_idrp_matricial(dap_atual, DIST_LINHA, DIST_ENTRELINHA, DIST_DIAG)
    vol_atual = np.exp(B0_VTCC + B1_VTCC * np.log(dap_atual) + B2_VTCC * np.log(ht_atual))

    LINHAS_atual, COLUNAS_atual = dap_atual.shape
    idade_atual = idade_inicial
    idade_final = idade_inicial + horizonte_meses
    
    grid_data_morte = np.full((LINHAS_atual, COLUNAS_atual), np.nan)
    mask_vizinhas_acumulado = np.zeros((LINHAS_atual, COLUNAS_atual), dtype=bool)
    dados_exportacao = []
    grid_contagem = np.zeros((LINHAS_atual, COLUNAS_atual))

    while idade_atual <= idade_final:
        if idade_atual in cronograma_mortalidade:
            perc_meta = cronograma_mortalidade[idade_atual]
            logit = B0_MORT + B1_MORT * idrp_atual
            prob_morte = 1 / (1 + np.exp(-logit))
            
            dap_flat = dap_atual.flatten()
            indices_vivos = np.where(~np.isnan(dap_flat))[0]
            n_morrer = int(len(indices_vivos) * perc_meta)
            
            if n_morrer > 0:
                prob_flat = prob_morte.flatten()
                probs_vivas = prob_flat[indices_vivos]
                soma = np.sum(probs_vivas)
                probs_norm = probs_vivas / soma if soma > 0 else None
                
                indices_sorteados = np.random.choice(indices_vivos, size=n_morrer, replace=False, p=probs_norm)
                linhas_m, colunas_m = np.unravel_index(indices_sorteados, (LINHAS_atual, COLUNAS_atual))
                
                dap_atual[linhas_m, colunas_m] = np.nan
                ht_atual[linhas_m, colunas_m] = np.nan
                vol_atual[linhas_m, colunas_m] = np.nan
                idrp_atual[linhas_m, colunas_m] = np.nan
                
                mask_new = np.isnan(grid_data_morte[linhas_m, colunas_m])
                l_new, c_new = linhas_m[mask_new], colunas_m[mask_new]
                grid_data_morte[l_new, c_new] = idade_atual
                
                for lm, cm in zip(linhas_m, colunas_m):
                    vizs = obter_vizinhos_8(lm, cm, LINHAS_atual, COLUNAS_atual)
                    for lv, cv in vizs:
                        if not np.isnan(dap_atual[lv, cv]):
                            grid_contagem[lv, cv] += 1

        grid_ile_dinamico = calcular_ile_dinamico(grid_data_morte, idade_atual, DIST_LINHA, DIST_ENTRELINHA, DIST_DIAG)
        mask_vizinhas_acumulado = mask_vizinhas_acumulado | (grid_ile_dinamico > 0)

        for l in range(LINHAS_atual):
            for c in range(COLUNAS_atual):
                status = "Intacta"
                if np.isnan(dap_atual[l, c]): status = "Morta"
                elif mask_vizinhas_acumulado[l, c]: status = "Vizinha"
                
                dados_exportacao.append({
                    "Idade": idade_atual, "Linha": l, "Coluna": c,
                    "DAP": dap_atual[l, c], "HT": ht_atual[l, c], "VOL": vol_atual[l, c],
                    "IID": idrp_atual[l, c], "Classe": status, 
                    "Vizinhos_Mortos": grid_contagem[l, c]
                })

        if idade_atual == idade_final: break
        passo = 12 if (idade_final - idade_atual) >= 12 else (idade_final - idade_atual)
        idade_futura = idade_atual + passo
        
        with np.errstate(invalid='ignore'):
            termo_idade = B1_DAP * ((idade_futura ** B2_DAP) - (idade_atual ** B2_DAP))
            termo_compensatorio = 0.0
            if beta_b3 != 0:
                termo_compensatorio = beta_b3 * grid_ile_dinamico
            
            expoente_total = termo_idade + termo_compensatorio
            dap_fut = dap_atual * np.exp(expoente_total)
            
            expoente_ht = B1_HT_PROJ * ((idade_futura ** B2_HT_PROJ) - (idade_atual ** B2_HT_PROJ))
            ht_fut = ht_atual * np.exp(expoente_ht)
        
        dap_fut = np.where(np.isnan(dap_atual), np.nan, dap_fut)
        ht_fut = np.where(np.isnan(ht_atual), np.nan, ht_fut)
        
        idrp_fut = calcular_idrp_matricial(dap_fut, DIST_LINHA, DIST_ENTRELINHA, DIST_DIAG)
        vol_fut = np.exp(B0_VTCC + B1_VTCC*np.log(dap_fut) + B2_VTCC*np.log(ht_fut))
        
        dap_atual, ht_atual, vol_atual, idrp_atual, idade_atual = dap_fut, ht_fut, vol_fut, idrp_fut, idade_futura

    return {
        "Volume_Final_Total": np.nansum(vol_atual),
        "Historico": pd.DataFrame(dados_exportacao),
        "Coeficientes_Usados": (beta_b3, 0)
    }

def otimizar_crescimento_compensatorio(floresta, idade_ini, horizonte, agenda_mortalidade):
    # Print removido para não poluir o terminal web
    res_100 = simular_floresta_compensatoria(floresta, idade_ini, horizonte, {}, beta_b3=0.0)
    meta_vol = res_100["Volume_Final_Total"]
    
    res_mort = simular_floresta_compensatoria(floresta, idade_ini, horizonte, agenda_mortalidade, beta_b3=0.0)
    
    def funcao_erro(b3_teste):
        res = simular_floresta_compensatoria(floresta, idade_ini, horizonte, agenda_mortalidade, beta_b3=b3_teste)
        return abs(res["Volume_Final_Total"] - meta_vol)
    
    opt = minimize_scalar(
        funcao_erro, 
        bracket=[0.0, 5.0], 
        method='brent',
        options={'maxiter': 2000, 'xtol': 1e-10}
    )
    b3_opt = opt.x
    
    res_comp = simular_floresta_compensatoria(floresta, idade_ini, horizonte, agenda_mortalidade, beta_b3=b3_opt)
    
    return {
        "Cenario_100": res_100,
        "Cenario_Mortalidade": res_mort,
        "Cenario_Compensatorio": res_comp,
        "Coeficientes_Otimos": (b3_opt, 0)
    }

def realizar_analise_anual_completa(df_sem_comp, df_com_comp, df_100_viva, coeficientes_otimos=None):
    """ Retorna as imagens com estética minimalista para Web """
    
    # 1. Configuração de Estética Clean (Fundo Branco, Sem Grades)
    sns.set_theme(style="white")
    
    viz_sem = df_sem_comp[df_sem_comp['Classe'] == 'Vizinha'].copy()
    viz_com = df_com_comp[df_com_comp['Classe'] == 'Vizinha'].copy()

    df_viz_total_check = pd.concat([viz_sem, viz_com])
    
    # Proteção caso não haja árvores mortas
    if df_viz_total_check.empty:
        fig_vazia, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Nenhuma mortalidade registrada no período.\nGráficos indisponíveis.", 
                ha='center', va='center', fontsize=12, color='gray')
        ax.axis('off')
        return fig_vazia, fig_vazia

    viz_sem['Cenario'] = 'Sem Compensação'
    viz_com['Cenario'] = 'Com Compensação'
    
    viz_sem['Centro_Classe'] = (viz_sem['DAP'] // 2 * 2 + 1).astype(int)
    viz_com['Centro_Classe'] = (viz_com['DAP'] // 2 * 2 + 1).astype(int)
    
    df_viz_total = pd.concat([viz_sem, viz_com])
    
    # Cores mais modernas para o fundo branco
    paleta_cores = {'Sem Compensação': '#e74c3c', 'Com Compensação': '#3498db'} 

    if not os.path.exists('Resultados_Graficos'):
        os.makedirs('Resultados_Graficos')

    # ==========================================
    # GRÁFICO 1: HISTOGRAMA DE TRANSIÇÃO
    # ==========================================
    idades_comuns = sorted(df_100_viva['Idade'].unique())
    idades_plot = [id for id in idades_comuns if id % 12 == 0 or id == idades_comuns[-1]]
    df_hist = df_viz_total[df_viz_total['Idade'].isin(idades_plot)]
    
    if df_hist.empty:
        fig_histograma, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Sem dados suficientes nos anos múltiplos de 12.", ha='center', va='center')
        ax.axis('off')
    else:
        ticks_exatos = sorted(df_hist['Centro_Classe'].unique())
        g = sns.FacetGrid(df_hist, col="Idade", col_wrap=3, height=4, aspect=1.3, sharey=False, sharex=False)
        g.map_dataframe(sns.histplot, x="Centro_Classe", hue="Cenario", discrete=True, multiple="dodge", shrink=0.8, 
                        palette=paleta_cores, edgecolor="black", linewidth=0.5, alpha=0.9)
        
        # Alteração 3: Nomes dos Eixos do Histograma
        g.set_axis_labels("Centro de Classe", "Número de Árvores")
        
        for ax in g.axes.flat:
            if len(ticks_exatos) > 0: ax.set_xticks(ticks_exatos)
            ax.grid(False) # Remove as grades internas
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cccccc')
            ax.spines['bottom'].set_color('#cccccc')
            
        g.add_legend(title="Situação")
        fig_histograma = g.fig
        plt.close(fig_histograma) 

    # ==========================================
    # GRÁFICO 2: EVOLUÇÃO DAP, HT e VOL
    # ==========================================
    fig_tendencia, axes = plt.subplots(1, 3, figsize=(18, 5))
    vars_plot = [('DAP', 'cm'), ('HT', 'm'), ('VOL', 'm³')]
    
    for i, (var, unit) in enumerate(vars_plot):
        sns.lineplot(data=df_viz_total, x='Idade', y=var, hue='Cenario', palette=paleta_cores, ax=axes[i], linewidth=3)
        axes[i].set_title(f"Evolução Média: {var}", fontsize=14, weight='bold', color='#333333', pad=15)
        axes[i].set_ylabel(f"{var} ({unit})", fontsize=12)
        axes[i].set_xlabel("Idade (meses)", fontsize=12)
        
        # Limpando o visual
        axes[i].grid(False)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_color('#cccccc')
        axes[i].spines['bottom'].set_color('#cccccc')
        
    plt.tight_layout()
    plt.close(fig_tendencia)

    return fig_histograma, fig_tendencia

def gerar_relatorio_individual_e_ganho(df_100, df_mort, df_comp):
    """ Retorna dicionário em vez de print """
    idade_final = df_comp['Idade'].max()
    
    vizinhas_comp = df_comp[(df_comp['Idade'] == idade_final) & (df_comp['Classe'] == 'Vizinha')]
    vizinhas_mort = df_mort[(df_mort['Idade'] == idade_final) & (df_mort['Classe'] == 'Vizinha')]
    
    dados_ganho = None
    if not vizinhas_comp.empty and not vizinhas_mort.empty:
        vol_comp = vizinhas_comp['VOL'].mean()
        vol_mort = vizinhas_mort['VOL'].mean()
        dap_comp = vizinhas_comp['DAP'].mean()
        dap_mort = vizinhas_mort['DAP'].mean()
        
        dados_ganho = {
            'ganho_vol': ((vol_comp / vol_mort) - 1) * 100,
            'vol_mort': vol_mort,
            'vol_comp': vol_comp,
            'ganho_dap': ((dap_comp / dap_mort) - 1) * 100,
            'dap_mort': dap_mort,
            'dap_comp': dap_comp
        }
        
    d1 = df_100.copy(); d1['Cenario'] = '100% Viva'
    d2 = df_mort.copy(); d2['Cenario'] = 'Com Mortalidade (Sem Compensação)'
    d3 = df_comp.copy(); d3['Cenario'] = 'Com Mortalidade (Com Compensação)'
    
    df_completo = pd.concat([d1, d2, d3], ignore_index=True)
    df_completo.insert(0, 'ID_Arvore', "L" + df_completo['Linha'].astype(str) + "_C" + df_completo['Coluna'].astype(str))
    
    if not os.path.exists('Resultados_Graficos'):
        os.makedirs('Resultados_Graficos')
        
    caminho_csv = "Resultados_Graficos/Historico_Individual_Arvores.csv"
    df_completo.to_csv(caminho_csv, index=False, sep=';', decimal=',')
    
    return dados_ganho, caminho_csv