import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import motor as mt 

st.set_page_config(page_title="ForestSIM", layout="wide", page_icon="🌲")

st.title("🌳 ForestSIM - Simulador Florestal 🌲")
st.markdown("""
**Universidade Federal de Viçosa (UFV) - Programa de Pós-Graduação em Ciência Florestal** 
""")
st.markdown("""
Ferramenta interativa para simulação espacial do crescimento de florestas, avaliando a **resposta compensatória** das árvores sobreviventes diante de diferentes cenários de mortalidade.
""")
st.divider()

# ==============================================================================
# BARRA LATERAL (INPUTS)
# ==============================================================================
st.sidebar.header("⚙️ Parâmetros da Floresta")

idade_ini = st.sidebar.number_input("Idade Inicial (meses)", min_value=12, max_value=60, value=24, step=12)
horizonte = st.sidebar.slider("Horizonte de Simulação (meses)", 12, 120, 60, step=12)

st.sidebar.subheader("🌱 Espaçamento Inicial")
col1, col2 = st.sidebar.columns(2)
dist_linha = col1.number_input("Linha (m)", value=3.0, step=0.1)
dist_entre = col2.number_input("Entrelinha (m)", value=2.0, step=0.1)

st.sidebar.subheader("📐 Dimensões da Malha")
col_l, col_c = st.sidebar.columns(2)
num_linhas = col_l.number_input("Nº de Linhas", min_value=5, max_value=150, value=33, step=1)
num_colunas = col_c.number_input("Nº de Colunas", min_value=5, max_value=150, value=33, step=1)

n_total_arvores = num_linhas * num_colunas
area_m2 = n_total_arvores * (dist_linha * dist_entre)
st.sidebar.info(f"**População Inicial:** {n_total_arvores} árvores\n\n**Área Simulada:** {area_m2:.1f} m²")

st.sidebar.subheader("🪾 Eventos de Mortalidade")
num_eventos = st.sidebar.number_input("Quantidade de Eventos", min_value=1, max_value=5, value=1, step=1)

agenda_usuario = {}
for i in range(num_eventos):
    c1, c2 = st.sidebar.columns(2)
    id_ev = c1.number_input(f"Idade {i+1} (m)", min_value=12, max_value=120, value=int(idade_ini) + (i*24), step=12, key=f"id_{i}")
    taxa_ev = c2.number_input(f"Morte (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key=f"tax_{i}")
    agenda_usuario[int(id_ev)] = taxa_ev / 100.0

st.sidebar.subheader("🎲 Padrão da Floresta")
st.sidebar.subheader("🎯 Resiliência (Fator Compensatório)")
modo_simulacao = st.sidebar.radio(
    "Modo de Simulação:", 
    ["Encontrar B3 Ideal (Otimizador)", "Definir B3 Manualmente"]
)

if modo_simulacao == "Definir B3 Manualmente":
    b3_usuario = st.sidebar.number_input("Valor de \u03B23 (\u00CDndice de Libera\u00E7\u00E3o)", min_value=0.0, max_value=2.0, value=0.0400, step=0.001, format="%.5f")
else:
    b3_usuario = None
travar_aleatoriedade = st.sidebar.checkbox("Manter o mesmo padrão", value=True)
semente_escolhida = st.sidebar.number_input("Código", value=42, step=1) if travar_aleatoriedade else None

with st.sidebar.expander("🔬 Parâmetros Biométricos (Modelos)", expanded=False):
    st.info("Altere os coeficientes para calibrar o simulador para o seu povoamento específico.")
    
    st.markdown("**1. Distribuição Inicial (Weibull 2P)**")
    st.latex(r"f(x) = \frac{\gamma}{\beta} \left(\frac{x}{\beta}\right)^{\gamma-1} \exp\left(-\left(\frac{x}{\beta}\right)^\gamma\right)")
    gamma_in = st.number_input("Gamma (Forma)", value=8.882151, format="%.6f")
    beta_in = st.number_input("Beta (Escala)", value=11.368466, format="%.6f")
    
    st.markdown("---")
    st.markdown("**2. Volume (Schumacher-Hall)**")
    st.latex(r"\ln(V) = \beta_0 + \beta_1 \ln(DAP) + \beta_2 \ln(HT)")
    b0_v_in = st.number_input("B0 Volume", value=-9.962793, format="%.6f")
    b1_v_in = st.number_input("B1 Volume", value=2.128458, format="%.6f")
    b2_v_in = st.number_input("B2 Volume", value=0.787242, format="%.6f")

    st.markdown("---")
    st.markdown("**3. Hipsométrica Inicial**")
    st.latex(r"\ln(HT) = \beta_0 + \beta_1 \ln(DAP)")
    b0_ht_in = st.number_input("B0 HT Inicial", value=-0.355914, format="%.6f")
    b1_ht_in = st.number_input("B1 HT Inicial", value=1.282668, format="%.6f")

    st.markdown("---")
    st.markdown("**4. Projeção de DAP (Modelo Base)**")
    st.latex(r"DAP_2 = DAP_1 \cdot \exp\left[\beta_1 (Idade_2^{\beta_2} - Idade_1^{\beta_2})\right]")
    b1_dap_in = st.number_input("B1 Proj. DAP", value=-9.533596, format="%.6f")
    b2_dap_in = st.number_input("B2 Proj. DAP", value=-0.812691, format="%.6f")

    st.markdown("---")
    st.markdown("**5. Projeção de HT**")
    st.latex(r"HT_2 = HT_1 \cdot \exp\left[\beta_1 (Idade_2^{\beta_2} - Idade_1^{\beta_2})\right]")
    b1_htp_in = st.number_input("B1 Proj. HT", value=-10.332914, format="%.6f")
    b2_htp_in = st.number_input("B2 Proj. HT", value=-0.626933, format="%.6f")

    st.markdown("---")
    st.markdown("**6. Risco de Mortalidade (Logística)**")
    st.latex(r"P(Morte) = \frac{1}{1 + \exp[-(\beta_0 + \beta_1 \cdot IDRP)]}")
    b0_mort_in = st.number_input("B0 Mortalidade", value=-0.8852, format="%.4f")
    b1_mort_in = st.number_input("B1 Mortalidade (Peso IDRP)", value=-6.1832, format="%.4f")

# ==============================================================================
# AÇÃO PRINCIPAL E INTEGRAÇÃO
# ==============================================================================
if st.button("🚀 Rodar Simulação Florestal", use_container_width=True, type="primary"):
    
    with st.spinner("Modelando matrizes espaciais e otimizando a resiliência..."):
        
        mt.IDADE_INICIAL = idade_ini
        mt.HORIZONTE = horizonte
        mt.DIST_LINHA = dist_linha
        mt.DIST_ENTRELINHA = dist_entre
        mt.DIST_DIAG = np.sqrt(dist_linha**2 + dist_entre**2)
        mt.AGENDA = agenda_usuario 
        mt.LINHAS = num_linhas
        mt.COLUNAS = num_colunas
        
        mt.GAMMA = gamma_in; mt.BETA = beta_in
        mt.B0_VTCC = b0_v_in; mt.B1_VTCC = b1_v_in; mt.B2_VTCC = b2_v_in
        mt.B0_HT = b0_ht_in; mt.B1_HT = b1_ht_in
        mt.B1_DAP = b1_dap_in; mt.B2_DAP = b2_dap_in
        mt.B1_HT_PROJ = b1_htp_in; mt.B2_HT_PROJ = b2_htp_in
        mt.B0_MORT = b0_mort_in; mt.B1_MORT = b1_mort_in

        floresta = mt.gerar_floresta_completa(mt.LINHAS, mt.COLUNAS, mt.BETA, mt.GAMMA, seed=semente_escolhida)
        resultados = mt.executar_simulacao_completa(floresta, mt.IDADE_INICIAL, mt.HORIZONTE, mt.AGENDA, beta_b3_usuario=b3_usuario, seed_simulacao=semente_escolhida)
        vol_100 = resultados['Cenario_100']['Volume_Final_Total']
        vol_mort = resultados['Cenario_Mortalidade']['Volume_Final_Total']
        vol_comp = resultados['Cenario_Compensatorio']['Volume_Final_Total']
        b3_opt = resultados['Coeficientes_Otimos'][0]
        
        df_100 = resultados['Cenario_100']['Historico']
        df_mort = resultados['Cenario_Mortalidade']['Historico']
        df_comp = resultados['Cenario_Compensatorio']['Historico']

        # ==============================================================================
        # DASHBOARD NA TELA
        # ==============================================================================
        # ==============================================================================
        # DASHBOARD NA TELA
        # ==============================================================================
        st.header("📊 1. Síntese de Produção")
        
        # Cálculos de Recuperação
        vol_perdido_total = vol_100 - vol_mort
        vol_recuperado = vol_comp - vol_mort
        
        # Evita divisão por zero caso a mortalidade seja zero
        if vol_perdido_total > 0:
            pct_recuperada = (vol_recuperado / vol_perdido_total) * 100
        else:
            pct_recuperada = 0.0
            
        m1, m2, m3 = st.columns(3)
        m1.metric("Meta Ideal (100% Viva)", f"{vol_100:.2f} m³")
        m2.metric("Real (Sem Compensação)", f"{vol_mort:.2f} m³", f"Perda de {vol_perdido_total:.2f} m³", delta_color="inverse")
        
        # Se for o otimizador, ele mostra cravado. Se for manual, mostra o quanto conseguiu recuperar
        if modo_simulacao == "Definir B3 Manualmente":
            m3.metric("Real (Com Compensação)", f"{vol_comp:.2f} m³", f"Recuperou {pct_recuperada:.1f}% da perda", delta_color="normal")
        else:
            m3.metric("Real (Com Compensação)", f"{vol_comp:.2f} m³", f"Recuperou 100.0% da perda", delta_color="normal")
        
        # --- ALTERAÇÃO 2: STATUS DA FLORESTA REMANESCENTE ---
        st.header("🌳 2. Demografia da Floresta Remanescente")
        
        idade_fim = df_comp['Idade'].max()
        df_final = df_comp[df_comp['Idade'] == idade_fim]
        total_pop = len(df_final)
        
        n_intactas = len(df_final[df_final['Classe'] == 'Intacta'])
        n_vizinhas = len(df_final[df_final['Classe'] == 'Vizinha'])
        n_mortas = len(df_final[df_final['Classe'] == 'Morta'])
        
        pct_intactas = (n_intactas / total_pop) * 100
        pct_vizinhas = (n_vizinhas / total_pop) * 100
        pct_mortas = (n_mortas / total_pop) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("🌳 Intactas (Sem falhas próximas)", f"{n_intactas} árvores", f"{pct_intactas:.1f}% do estande", delta_color="off")
        c2.metric("🎯 Vizinhas (Com falhas próximas)", f"{n_vizinhas} árvores", f"{pct_vizinhas:.1f}% do estande", delta_color="off")
        c3.metric("💀 Mortas (Abertura de Dossel)", f"{n_mortas} falhas", f"{pct_mortas:.1f}% do estande", delta_color="off")

        # --- ESFORÇO DE CRESCIMENTO ---
        st.header("🎯 3. Esforço de Crescimento (Fator de Compensação: {:.12f})".format(b3_opt))
        
        st.markdown("A projeção do diâmetro das árvores vizinhas foi alterada pela inclusão do termo aditivo compensatório:")
        st.latex(r"DAP_2 = DAP_1 \cdot \exp\left[\beta_1 (Idade_2^{\beta_2} - Idade_1^{\beta_2}) + \beta_3 \cdot ILED\right]")
        st.markdown("Nesta formulação, o parâmetro $\beta_3$ atua como um multiplicador de resiliência sobre o **Índice de Liberação Espacial Dinâmico ($ILED$)**, capitalizando o espaço extra deixado pelas clareiras e o tempo decorrido desde a mortalidade.")
        
        dados_ganho, caminho_csv = mt.gerar_relatorio_individual_e_ganho(df_100, df_mort, df_comp)
        
        if dados_ganho:
            st.info(f"""
            Para atingir a resiliência produtiva do talhão, as **árvores vizinhas sobreviventes** precisaram crescer em média:
            * **+ {dados_ganho['ganho_vol']:.2f}% em Volume** (Média subiu de {dados_ganho['vol_mort']:.4f} m³ para {dados_ganho['vol_comp']:.4f} m³).
            * **+ {dados_ganho['ganho_dap']:.2f}% em Diâmetro (DAP)** (Média subiu de {dados_ganho['dap_mort']:.2f} cm para {dados_ganho['dap_comp']:.2f} cm).
            """)

        # --- ALTERAÇÃO 4: GRÁFICOS REDESENHADOS ---
        st.header("📈 4. Evolução Dendrométrica das Vizinhas")
        fig_hist, fig_tend = mt.realizar_analise_anual_completa(df_mort, df_comp, df_100, coeficientes_otimos=(b3_opt, 0))
        st.pyplot(fig_tend)
        plt.close(fig_tend)

        st.subheader("Transição de Classes Diamétricas")
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        # --- MAPAS ---
        st.header("🗺️ 5. Matriz Espacial e Cicatrizes de Mortalidade")
        
        grid_status = np.zeros((mt.LINHAS, mt.COLUNAS)) 
        grid_viz_mortos = np.zeros((mt.LINHAS, mt.COLUNAS))

        for _, row in df_final.iterrows():
            l, c = int(row['Linha']), int(row['Coluna'])
            grid_viz_mortos[l, c] = row['Vizinhos_Mortos']
            if row['Classe'] == 'Morta': grid_status[l, c] = 2
            elif row['Classe'] == 'Vizinha': grid_status[l, c] = 1
            else: grid_status[l, c] = 0

        fig_map, ax_map = plt.subplots(1, 2, figsize=(14, 6))
        
        cmap_status = ListedColormap(['forestgreen', '#3498db', '#e74c3c']) # Cores consistentes com o grafico
        sns.heatmap(grid_status, cmap=cmap_status, cbar=False, ax=ax_map[0], square=True, linewidths=0.5, linecolor='white', xticklabels=False, yticklabels=False)
        ax_map[0].set_title(f"Status Final ({idade_fim} meses)", fontsize=14)
        legend_elements = [Patch(facecolor='#e74c3c', label='Morta'),
                           Patch(facecolor='#3498db', label='Vizinha (Compensando)'),
                           Patch(facecolor='forestgreen', label='Intacta')]
        ax_map[0].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

        sns.heatmap(grid_viz_mortos, cmap='Blues', annot=False, ax=ax_map[1], square=True, xticklabels=False, yticklabels=False)
        ax_map[1].set_title("Intensidade de Exposição (Nº Vizinhos Mortos)", fontsize=14)
        
        plt.tight_layout()
        st.pyplot(fig_map)
        plt.close(fig_map)
        
        st.markdown("---")
        
        st.subheader("📥 Exportação de Dados")
        with open(caminho_csv, "rb") as file:
            st.download_button(
                label="Baixar Painel Longitudinal Completo (.CSV)",
                data=file,
                file_name="Historico_Individual_Arvores.csv",
                mime="text/csv"
            )