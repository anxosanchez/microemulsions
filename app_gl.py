import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize

# --- 1. BASE DE DATOS GLOBAL ---
DATA = {
    "Solventes (Fase Oleosa)": {
        "Sunflowerato de Metilo": {"hsp": [16.2, 3.2, 3.8], "rho": 880, "eacn": 1.5, "fp": 170, "price": 1.65, "ghs": []},
        "Soyato de Metilo": {"hsp": [16.1, 3.1, 3.7], "rho": 885, "eacn": 1.4, "fp": 175, "price": 1.60, "ghs": []},
        "Palmitato de Metilo": {"hsp": [16.3, 3.3, 3.9], "rho": 870, "eacn": 1.8, "fp": 180, "price": 1.55, "ghs": []},
        "DBE (Ésteres Dibásicos)": {"hsp": [16.5, 7.5, 7.0], "rho": 1060, "eacn": -5.4, "fp": 108, "price": 2.85, "ghs": ["H319", "P264", "P280", "P305+P351+P338"]},
        "Tolueno": {"hsp": [18.0, 1.4, 2.0], "rho": 867, "eacn": 1.0, "fp": 4, "price": 0.95, "ghs": ["H225", "H304", "H315", "H336", "H361d", "P210", "P260", "P280", "P301+P310", "P331"]},
        "Acetato de Butilo": {"hsp": [15.8, 3.7, 6.3], "rho": 881, "eacn": -2.5, "fp": 22, "price": 1.25, "ghs": ["H226", "H336", "P210", "P261", "P304+P340"]},
    },
    "Cosolventes": {
        "Ningún": {"hsp": [0, 0, 0], "rho": 1.0, "f_hld": 0, "fp": 200, "price": 0, "ghs": []},
        "Alcohol Bencílico": {"hsp": [18.4, 6.3, 13.7], "rho": 1045, "f_hld": -0.2, "fp": 93, "price": 2.45, "ghs": ["H302", "H332", "P261", "P264", "P270"]},
        "Carbonato de Propileno": {"hsp": [20.0, 18.0, 4.1], "rho": 1200, "f_hld": 0.1, "fp": 132, "price": 2.10, "ghs": ["H319", "P264", "P280"]},
        "Glicerina": {"hsp": [17.4, 12.1, 29.3], "rho": 1260, "f_hld": -0.8, "fp": 160, "price": 0.95, "ghs": []},
        "Butil Diglicol": {"hsp": [16.0, 7.0, 10.6], "rho": 953, "f_hld": -0.2, "fp": 105, "price": 3.20, "ghs": ["H319", "P264", "P280"]},
        "Cyrene": {"hsp": [18.9, 12.4, 7.1], "rho": 1250, "f_hld": 0.4, "fp": 108, "price": 8.50, "ghs": ["H319", "P264", "P280"]},
        "Acetona": {"hsp": [15.5, 10.4, 7.0], "rho": 784, "f_hld": 0.0, "fp": -20, "price": 0.85, "ghs": ["H225", "H319", "H336", "P210", "P261", "P305+P351+P338"]},
        "Loxanol (Xenérico)": {"hsp": [16.5, 4.2, 8.5], "rho": 980, "f_hld": -0.1, "fp": 105, "price": 4.20, "ghs": []},
    },
    "Tansioactivos": {
        "APG (Non Iónico)": {"cc": 1.5, "hlb": 13.5, "rho": 1100, "hsp": [18.0, 12.0, 15.0], "price": 4.50, "ghs": ["H318", "P280", "P305+P351+P338"], "type": "NI", "fp": 200},
        "SLES (Aniónico)": {"cc": -2.0, "hlb": 40.0, "rho": 1050, "hsp": [17.5, 11.0, 9.5], "price": 3.85, "ghs": ["H315", "H318", "P264", "P280", "P302+P352"], "type": "A", "fp": 200},
        "Rokawin NL7": {"cc": 1.2, "hlb": 12.0, "rho": 1010, "hsp": [16.5, 6.0, 10.0], "price": 3.10, "ghs": ["H302", "H318", "P264", "P280", "P301+P312"], "type": "NI", "fp": 150},
        "Kolliphor P188": {"cc": 3.5, "hlb": 29.0, "rho": 1060, "hsp": [17.0, 8.0, 12.0], "price": 12.50, "ghs": [], "type": "NI", "fp": 250},
        "Tween 80": {"cc": 2.2, "hlb": 15.0, "rho": 1070, "hsp": [17.5, 10.5, 14.0], "price": 5.80, "ghs": [], "type": "NI", "fp": 113},
    },
    "Cotansioactivos (Alcohois)": {
        "Etanol": {"hsp": [15.8, 8.8, 19.4], "rho": 789, "fp": 13, "price": 1.20, "ghs": ["H225", "H319", "P210", "P233", "P280"]},
        "Butanol": {"hsp": [16.0, 5.7, 15.8], "rho": 810, "fp": 35, "price": 1.80, "ghs": ["H226", "H302", "H315", "H318", "H335", "P210", "P261", "P280"]},
        "Isopropanol (IPA)": {"hsp": [15.8, 6.1, 16.4], "rho": 786, "fp": 12, "price": 1.50, "ghs": ["H225", "H319", "H336", "P210", "P261", "P280"]},
    },
    "Resinas (Obxectivos)": {
        "Alquidica": {"hsp": [18.5, 4.5, 5.1], "r0": 8.0},
        "Nitrocelulosa": {"hsp": [15.4, 10.1, 8.8], "r0": 11.5},
        "Poliuretano": {"hsp": [17.8, 10.5, 11.2], "r0": 9.0},
        "PVC": {"hsp": [18.8, 9.2, 3.5], "r0": 7.5},
        "Acrílica": {"hsp": [18.5, 9.0, 7.5], "r0": 9.2},
        "Caucho Clorado": {"hsp": [18.0, 6.0, 7.0], "r0": 8.5},
        "Sintética": {"hsp": [17.5, 3.5, 3.0], "r0": 7.0},
        "Epoxi-Poliamida": {"hsp": [18.5, 9.5, 10.0], "r0": 11.0},
    }
}

GHS_MAP = {
    # H-Phrases
    "H225": {"text": "Líquido e vapores moi inflamables", "type": "Flame", "cat": "H"},
    "H226": {"text": "Líquido e vapores inflamables", "type": "Flame", "cat": "H"},
    "H302": {"text": "Nocivo en caso de inxestión", "type": "Exclamation", "cat": "H"},
    "H304": {"text": "Pode ser mortal en caso de inxestión e penetración nas vías respiratorias", "type": "Health", "cat": "H"},
    "H315": {"text": "Provoca irritación cutánea", "type": "Exclamation", "cat": "H"},
    "H318": {"text": "Provoca lesións oculares graves", "type": "Corrosive", "cat": "H"},
    "H319": {"text": "Provoca irritación ocular grave", "type": "Exclamation", "cat": "H"},
    "H332": {"text": "Nocivo en caso de inhalación", "type": "Exclamation", "cat": "H"},
    "H335": {"text": "Pode irritar as vías respiratorias", "type": "Exclamation", "cat": "H"},
    "H336": {"text": "Pode provocar somnolencia ou vertixe", "type": "Exclamation", "cat": "H"},
    "H361d": {"text": "Sospéitase que dana ao feto", "type": "Health", "cat": "H"},
    # P-Phrases
    "P210": {"text": "Manter lonxe do calor, superficies calentes, chispas, chamas abertas e calquera outra fonte de ignición. Non fumar.", "cat": "P"},
    "P233": {"text": "Manter o recipiente hermeticamente pechado.", "cat": "P"},
    "P260": {"text": "Non respirar o po/o fume/o gas/a néboa/os vapores/o aerosol.", "cat": "P"},
    "P261": {"text": "Evitar respirar o po/o fume/o gas/a néboa/os vapores/o aerosol.", "cat": "P"},
    "P264": {"text": "Lavarse as mans conciudamente tras a manipulación.", "cat": "P"},
    "P270": {"text": "Non comer, beber nin fumar durante a súa utilización.", "cat": "P"},
    "P280": {"text": "Llevar luvas/prendas/gafas/máscara de protección.", "cat": "P"},
    "P301+P310": {"text": "EN CASO DE INXESTIÓN: Chamar inmediatamente a un CENTRO DE INFORMACIÓN TOXICOLÓXICA ou a un médico.", "cat": "P"},
    "P301+P312": {"text": "EN CASO DE INXESTIÓN: Chamar a un CENTRO DE INFORMACIÓN TOXICOLÓXICA ou a un médico se a persoa se encontra mal.", "cat": "P"},
    "P302+P352": {"text": "EN CASO DE CONTACTO COA PEL: Lavar con abundante auga.", "cat": "P"},
    "P304+P340": {"text": "EN CASO DE INHALACIÓN: Transportar á persoa ao aire libre e mantela nunha posición que facilite a respiración.", "cat": "P"},
    "P305+P351+P338": {"text": "EN CASO DE CONTACTO COS OLLOS: Enxaugar con coidado con auga durante varios minutos. Quitar as lentes de contacto se leva e resulta fácil. Seguir enxaugando.", "cat": "P"},
    "P331": {"text": "NON provocar o vómito.", "cat": "P"}
}

PICTOGRAMS = {
    "Flame": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/GHS-pictogram-flamme.svg/100px-GHS-pictogram-flamme.svg.png",
    "Exclamation": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/GHS-pictogram-exclam.svg/100px-GHS-pictogram-exclam.svg.png",
    "Health": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/GHS-pictogram-silhouete.svg/100px-GHS-pictogram-silhouete.svg.png",
    "Corrosive": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/GHS-pictogram-acid.svg/100px-GHS-pictogram-acid.svg.png",
    "Toxic": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/GHS-pictogram-skull.svg/100px-GHS-pictogram-skull.svg.png",
    "Environment": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/GHS-pictogram-pollut.svg/100px-GHS-pictogram-pollut.svg.png"
}

# --- 2. FUNCIÓNS LÓXICAS ---

def calculate_hld(sf, oil_eacn, salinity, temp=25):
    """Modelo de Diferencia Hidrofílica-Lipofílica."""
    k = 0.17
    s = salinity if salinity > 0 else 0.001
    if sf.get("type") == "A":  # Aniónico
        return np.log(s) - k * oil_eacn + sf["cc"]
    else:  # Non Iónico
        b = 0.13
        return b * s - k * oil_eacn + sf["cc"]

def calculate_red(h_mix, resin):
    """Diferenza de Enerxía Relativa para a solubilidade."""
    ra = np.sqrt(4 * (h_mix[0] - resin["hsp"][0])**2 + 
                 (h_mix[1] - resin["hsp"][1])**2 + 
                 (h_mix[2] - resin["hsp"][2])**2)
    return ra / resin["r0"]

def get_physical_properties(ws, rhos, fps):
    """Estimar propiedades físicas da mestura."""
    rho_mix = 1 / np.sum(ws / rhos)
    log_fp = np.sum(ws * np.log10(fps))
    fp_mix = 10**log_fp
    return rho_mix, fp_mix

# --- 3. CONFIGURACIÓN DA IU ---
st.set_page_config(page_title="MicroSaaS Pro | Panel de Laboratorio", layout="wide", page_icon="🧪")

# CSS personalizado para aspecto de laboratorio escuro premium
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    .main { background-color: #0f172a; padding: 2rem; }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; }
    .stMetric { 
        background-color: #1e293b !important; padding: 20px !important; 
        border-radius: 12px !important; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3) !important;
        border: 1px solid #334155 !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: #0f172a; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: transparent !important;
        color: #94a3b8 !important; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom-color: #38bdf8 !important; }
    [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
    [data-testid="stSidebar"] .stMarkdown h1, h2, h3 { color: #f1f5f9; }
    h1, h2, h3 { color: #f8fafc; font-weight: 800; }
    .stTable, .stDataFrame { background-color: #1e293b; color: #f1f5f9; border-radius: 8px; }
    div[data-testid="stExpander"] { background-color: #1e293b !important; border: 1px solid #334155 !important; }
    .stAlert { background-color: #1e293b !important; color: #f1f5f9 !important; border: 1px solid #334155 !important; }
    div[data-testid="stJson"] { background-color: #1e293b !important; padding: 1rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Suite de Fabricación de Microemulsións Industriais")
st.markdown("---")

with st.sidebar:
    st.header("🛠️ Deseñador de Formulación")
    
    with st.expander("Selección de Compoñentes", expanded=True):
        res_k = st.selectbox("Resina Obxectivo", list(DATA["Resinas (Obxectivos)"].keys()))
        sol_k = st.selectbox("Solvente Base", list(DATA["Solventes (Fase Oleosa)"].keys()))
        cos_k = st.selectbox("Cosolvente de Nivelación", list(DATA["Cosolventes"].keys()))
        sur_k = st.selectbox("Tansioactivo Principal", list(DATA["Tansioactivos"].keys()))
        alc_k = st.selectbox("Cotansioactivo (Alcohol)", list(DATA["Cotansioactivos (Alcohois)"].keys()))
    
    with st.expander("Concentracións (%)", expanded=True):
        p_sol = st.slider("% Fase Solvente", 5, 60, 25)
        p_cos = st.slider("% Cosolvente", 0, 15, 5)
        p_sur = st.slider("% Tansioactivo", 5, 30, 12)
        p_alc = st.slider("% Cotansioactivo", 2, 20, 8)
        p_wat = 100 - p_sol - p_cos - p_sur - p_alc
        
        if p_wat < 5:
            st.error("Demasiada fase orgánica! A auga debe ser polo menos o 5%.")
            p_wat = 5
        st.info(f"💧 Fase Acuosa: {p_wat}%")
        
        salinity = st.number_input("Salinidade (% NaCl na mestura)", 0.0, 10.0, 0.5, step=0.1)

# Preparación de Cálculos
S, C, Sf, A, R = DATA["Solventes (Fase Oleosa)"][sol_k], DATA["Cosolventes"][cos_k], \
                 DATA["Tansioactivos"][sur_k], DATA["Cotansioactivos (Alcohois)"][alc_k], \
                 DATA["Resinas (Obxectivos)"][res_k]

W = {"hsp": [15.5, 16.0, 42.1], "rho": 997, "fp": 1000, "price": 0.01}
components = [S, C, Sf, A, W]
ws = np.array([p_sol, p_cos, p_sur, p_alc, p_wat]) / 100
rhos = np.array([c["rho"] for c in components])
fps = np.array([c.get("fp", 200) for c in components])
hsps = np.array([c["hsp"] for c in components])

rho_mix, fp_mix = get_physical_properties(ws, rhos, fps)
v_fractions = (ws / rhos) / np.sum(ws / rhos)
h_mix = np.dot(v_fractions, hsps)

red = calculate_red(h_mix, R)
oil_eacn = (p_sol * S["eacn"] + p_cos * C.get("f_hld", 0)) / (p_sol + p_cos) if (p_sol + p_cos) > 0 else 0
hld = calculate_hld(Sf, oil_eacn, salinity)

# --- 4. SECCIÓNS DO PANEL ---
t1, t2, t3 = st.tabs(["📊 Análise de Laboratorio", "🛡️ Seguridade e Normativa", "🏭 Fabricación"])

with t1:
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.subheader("Espazo de Solubilidade de Hansen (3D)")
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x_s = R['r0'] * np.cos(u) * np.sin(v) + R['hsp'][0]
        y_s = R['r0'] * np.sin(u) * np.sin(v) + R['hsp'][1]
        z_s = R['r0'] * np.cos(v) + R['hsp'][2]
        fig = go.Figure()
        fig.add_trace(go.Surface(x=x_s, y=y_s, z=z_s, opacity=0.4, showscale=False,
            colorscale=[[0, '#38bdf8'], [1, '#38bdf8']],
            contours=dict(x=dict(show=True, color="white", width=1), y=dict(show=True, color="white", width=1), z=dict(show=True, color="white", width=1)),
            name=f"Obxectivo: {res_k}"))
        fig.add_trace(go.Scatter3d(x=[h_mix[0]], y=[h_mix[1]], z=[h_mix[2]], mode='markers', 
            marker=dict(size=12, color='#ef4444', symbol='diamond', line=dict(color='white', width=2)), name="Formulación Actual"))
        fig.add_trace(go.Scatter3d(x=[R['hsp'][0]], y=[R['hsp'][1]], z=[R['hsp'][2]], mode='markers', 
            marker=dict(size=6, color='#fbbf24'), name=f"Centro de {res_k}"))
        center, r_val = R['hsp'], R['r0'] + 5
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            scene=dict(xaxis=dict(title='δD', range=[center[0]-r_val, center[0]+r_val], gridcolor="#475569"),
                       yaxis=dict(title='δP', range=[center[1]-r_val, center[1]+r_val], gridcolor="#475569"),
                       zaxis=dict(title='δH', range=[center[2]-r_val, center[2]+r_val], gridcolor="#475569"),
                       aspectmode='cube', bgcolor="#0f172a"),
            margin=dict(l=0, r=0, b=0, t=0), legend=dict(x=0.02, y=0.98, font=dict(color="white"), bgcolor="rgba(15, 23, 42, 0.8)"))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Indicadores de Rendemento")
        m1, m2 = st.columns(2)
        m1.metric("RED de Solubilidade", f"{red:.2f}", delta="-0.05" if red < 1 else "+0.05", delta_color="inverse")
        m2.metric("HLD de Estabilidade", f"{hld:.2f}", delta="Óptimo" if abs(hld) < 0.5 else "Inestable")
        st.markdown("### Perfil Físico")
        tds_data = {"Densidade": f"{rho_mix:.1f} kg/m³", "Punto de Inflamación": f"{fp_mix:.1f} °C",
                    "Estabilidade": "✅ Alta" if abs(hld) < 0.5 else "⚠️ Media" if abs(hld) < 1.0 else "❌ Baixa",
                    "Solubilidade": "✅ Excelente" if red < 1 else "⚠️ Parcial" if red < 1.5 else "❌ Pobre"}
        st.json(tds_data)
        if st.button("🚀 Optimizador: Minimizar RED e HLD"):
            with st.spinner("Calculando proporcións óptimas..."):
                def obj(x):
                    p_wat_opt = 100 - x[0] - p_cos - x[1] - x[2]
                    if p_wat_opt < 10: return 1e6
                    ws_opt = np.array([x[0], p_cos, x[1], x[2], p_wat_opt]) / 100
                    v_opt = (ws_opt / rhos) / np.sum(ws_opt / rhos)
                    h_opt = np.dot(v_opt, hsps)
                    r_opt = calculate_red(h_opt, R)
                    o_eacn = (x[0] * S["eacn"] + p_cos * C.get("f_hld", 0)) / (x[0] + p_cos)
                    h_opt_val = calculate_hld(Sf, o_eacn, x[3])
                    return 10*h_opt_val**2 + 5*r_opt**2 + 0.1*x[1]
                res = minimize(obj, [p_sol, p_sur, p_alc, salinity], bounds=[(10, 50), (5, 25), (2, 15), (0, 5)])
                if res.success:
                    st.success(f"Optimizado! Solvent: {res.x[0]:.1f}%, Tansioactivo: {res.x[1]:.1f}%, Alcohol: {res.x[2]:.1f}%, Salinidade: {res.x[3]:.1f}%")
                else: st.warning("Non converxeu. Probe outro punto de inicio.")

with t2:
    st.subheader("Cumprimento Normativo e GHS")
    relevant_h, relevant_p, pictos_to_show = [], [], set()
    check_list = [(sol_k, S, p_sol/100), (cos_k, C, p_cos/100), (sur_k, Sf, p_sur/100), (alc_k, A, p_alc/100)]
    for name, comp, weight in check_list:
        if weight > 0.05:
            for code in comp.get("ghs", []):
                if code in GHS_MAP:
                    data = GHS_MAP[code]
                    if data["cat"] == "H":
                        relevant_h.append({"Código": code, "Indicación": data["text"], "Orixe": name})
                        if "type" in data: pictos_to_show.add(data["type"])
                    else: relevant_p.append({"Código": code, "Precaución": data["text"], "Orixe": name})

    if pictos_to_show:
        pic_cols = st.columns(len(pictos_to_show))
        for i, p_type in enumerate(pictos_to_show):
            with pic_cols[i]: st.image(PICTOGRAMS[p_type], width=80); st.caption(f"GHS {p_type}")

    if relevant_h: st.markdown("### ⚠️ Indicacións de Perigo (Frases H)"); st.table(pd.DataFrame(relevant_h))
    if relevant_p: st.markdown("### 🛡️ Consellos de Prudencia (Frases P)"); st.table(pd.DataFrame(relevant_p))

    st.markdown("---")
    st.markdown("### 🔥 Clasificación de Inflamabilidade CLP/ECHA")
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**Punto de Inflamación calculado:** {fp_mix:.1f} °C")
        if fp_mix < 23: st.error("🚨 Categoría CLP 2: Moi Inflamable"); st.caption("Punto de inflamación < 23°C")
        elif 23 <= fp_mix <= 60: st.warning("⚠️ Categoría CLP 3: Inflamable"); st.caption("23°C ≤ Punto infl. ≤ 60°C")
        else: st.success("✅ Non clasificado por inflamabilidade"); st.caption("Punto de inflamación > 60°C (Regulamento CLP)")
    with c2:
        voc = "Si" if S["fp"] < 250 else "Biolóxico/Baixo COV"
        st.write(f"**Estado de COV (2004/42/CE):** {voc}")

with t3:
    st.header("🏭 Calculadora de Produción por Lotes")
    col_batch1, col_batch2 = st.columns(2)
    with col_batch1:
        batch_vol = st.number_input("Volume do lote (Litros)", 100, 50000, 1000)
        brine_conc = st.slider("Concentración de salmoira de stock (% NaCl)", 5.0, 25.0, 15.0)
    total_kg = batch_vol * (rho_mix / 1000)
    cost_total = np.sum(ws * np.array([c.get("price", 0) for c in components])) * total_kg
    with col_batch2:
        st.metric("Masa Total do Lote", f"{total_kg:.1f} kg")
        st.metric("Custo Estimado de Materiais", f"€{cost_total:.2f}")

    st.markdown("### 📋 Instrucións de Pesada")
    w_sol, w_cos, w_sur, w_alc = total_kg * (p_sol/100), total_kg * (p_cos/100), total_kg * (p_sur/100), total_kg * (p_alc/100)
    target_salt = total_kg * (salinity / 100)
    w_brine = target_salt / (brine_conc / 100)
    w_water = (total_kg * (p_wat / 100)) - w_brine
    if w_water < 0: st.error("Erro: A salmoira de stock está demasiado diluída.")
    production_df = pd.DataFrame({"Compoñente": [sol_k, cos_k, sur_k, alc_k, "Salmoira de Stock", "Auga Pura"],
                                  "Peso (kg)": [w_sol, w_cos, w_sur, w_alc, w_brine, w_water], "Paso": [1, 2, 3, 4, 5, 5]})
    st.table(production_df)
    st.success("💡 **Consello de mestura:** Mesture primeiro a fase orgánica (pasos 1-4), despois engada a fase acuosa (paso 5) con axitación persistente.")

    st.markdown("---")
    st.subheader("📄 Xeración de Informe")
    report_text = f"""FICHA TÉCNICA
Proxecto: Lote de Microemulsión - {sol_k}
--------------------------------------
Propiedades Físicas:
- Densidade: {rho_mix:.1f} kg/m3
- Punto de Inflamación: {fp_mix:.1f} C
- Estabilidade (HLD): {hld:.2f}
- Solubilidade (RED vs {res_k}): {red:.2f}

Formulación:
- {sol_k}: {p_sol}%
- {cos_k}: {p_cos}%
- {sur_k}: {p_sur}%
- {alc_k}: {p_alc}%
- Auga: {p_wat}%
"""
    st.download_button("📥 Descargar Ficha Técnica (TXT)", report_text, file_name="TDS_Microemulsion_GL.txt")

with st.expander("📚 Datos científicos de referencia"):
    for cat, items in DATA.items():
        st.write(f"**{cat}**")
        st.dataframe(pd.DataFrame(items).T)
