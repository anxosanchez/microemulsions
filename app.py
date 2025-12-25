import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize

# --- 1. TRANSLATION DATA ---
LANGS = {
    "English": {
        "title": "🧪 Industrial Microemulsion Manufacturing Suite",
        "designer": "🛠️ Formulation Designer",
        "selection": "Component Selection",
        "resin": "Target Resin",
        "solvent": "Base Solvent",
        "cosolvent": "Leveling Cosolvent",
        "surfactant": "Primary Surfactant",
        "cosurfactant": "Cosurfactant (Alcohol)",
        "concs": "Concentrations (%)",
        "p_sol": "% Solvent Phase",
        "p_cos": "% Cosolvent",
        "p_sur": "% Surfactant",
        "p_alc": "% Cosurfactant",
        "wat_err": "Too much organic phase! Water must be at least 5%.",
        "wat_info": "💧 Aqueous Phase: ",
        "salinity": "Salinity (% NaCl in mix)",
        "tabs": ["📊 Lab Analysis", "🛡️ Regulatory & Safety", "🏭 Manufacturing"],
        "hsp_title": "Hansen Solubility Space (3D)",
        "target": "Target",
        "current": "Current Formulation",
        "center": "Center",
        "perf": "Performance Indicators",
        "red": "Solubility RED",
        "hld": "Stability HLD",
        "phys": "Physical Profile",
        "density": "Density",
        "fp": "Flash Point",
        "stability": "Stability",
        "solubility": "Solubility",
        "optimal": "Optimal",
        "unstable": "Unstable",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        "excellent": "Excellent",
        "partial": "Partial",
        "poor": "Poor",
        "opt_btn": "🚀 Optimizer: Minimize RED & HLD",
        "opt_wait": "Calculating optimal ratios...",
        "opt_succ": "Optimized! Set Oil: ",
        "opt_err": "Could not converge. Try different baseline.",
        "reg_title": "Regulatory Compliance & GHS",
        "h_phrases": "⚠️ Hazard Statements (H-Phrases)",
        "p_phrases": "🛡️ Precautionary Statements (P-Phrases)",
        "flam_title": "🔥 CLP/ECHA Flammability Classification",
        "calc_fp": "Calculated FP",
        "clp2": "🚨 CLP Category 2: Highly Flammable",
        "clp3": "⚠️ CLP Category 3: Flammable",
        "non_flam": "✅ Not Classified for Flammability",
        "clp_reg": "CLP Regulation",
        "voc": "VOC Status (2004/42/EC)",
        "bio_voc": "Biological/Low VOC",
        "batch_title": "🏭 Batch Production Calculator",
        "batch_vol": "Batch Volume (Liters)",
        "brine_conc": "Stock Brine Concentration (% NaCl)",
        "total_mass": "Total Batch Mass",
        "total_cost": "Estimated Material Cost",
        "weighing": "📋 Weighing Instructions",
        "comp": "Component",
        "weight": "Weight (kg)",
        "step": "Step",
        "brine_stock": "Brine Stock",
        "pure_water": "Pure Water",
        "brine_err": "Error: Brine stock is too diluted.",
        "tip": "💡 **Mixing Tip:** Blend organic phase (Steps 1-4) first, then slowly add aqueous phase (Step 5) under high shear.",
        "report_title": "📄 Report Generation",
        "tds_btn": "📥 Download Technical Data Sheet (TXT)",
        "ref_data": "📚 Scientific Reference Data",
        "origin": "Origin",
        "code": "Code",
        "statement": "Statement",
        "precaution": "Precaution"
    },
    "Español": {
        "title": "🧪 Suite de Fabricación de Microemulsiones Industriales",
        "designer": "🛠️ Diseñador de Formulación",
        "selection": "Selección de Componentes",
        "resin": "Resina Objetivo",
        "solvent": "Solvente Base",
        "cosolvent": "Cosolvente de Nivelación",
        "surfactant": "Tensioactivo Principal",
        "cosurfactant": "Cotensioactivo (Alcohol)",
        "concs": "Concentraciones (%)",
        "p_sol": "% Fase Solvente",
        "p_cos": "% Cosolvente",
        "p_sur": "% Tensioactivo",
        "p_alc": "% Cotensioactivo",
        "wat_err": "¡Demasiada fase orgánica! El agua debe ser al menos el 5%.",
        "wat_info": "💧 Fase Acuosa: ",
        "salinity": "Salinidad (% NaCl en mezcla)",
        "tabs": ["📊 Análisis de Laboratorio", "🛡️ Seguridad y Normativa", "🏭 Fabricación"],
        "hsp_title": "Espacio de Solubilidad de Hansen (3D)",
        "target": "Objetivo",
        "current": "Formulación Actual",
        "center": "Centro",
        "perf": "Indicadores de Rendimiento",
        "red": "RED de Solubilidad",
        "hld": "HLD de Estabilidad",
        "phys": "Perfil Físico",
        "density": "Densidad",
        "fp": "Punto de Inflamación",
        "stability": "Estabilidad",
        "solubility": "Solubilidad",
        "optimal": "Óptimo",
        "unstable": "Inestable",
        "high": "Alta",
        "medium": "Media",
        "low": "Baja",
        "excellent": "Excelente",
        "partial": "Parcial",
        "poor": "Pobre",
        "opt_btn": "🚀 Optimizador: Minimizar RED y HLD",
        "opt_wait": "Calculando proporciones óptimas...",
        "opt_succ": "¡Optimizado! Solvente: ",
        "opt_err": "No convergió. Pruebe otro punto de inicio.",
        "reg_title": "Cumplimiento Normativo y GHS",
        "h_phrases": "⚠️ Indicaciones de Peligro (Frases H)",
        "p_phrases": "🛡️ Consejos de Prudencia (Frases P)",
        "flam_title": "🔥 Clasificación de Inflamabilidad CLP/ECHA",
        "calc_fp": "Punto de Inflamación calculado",
        "clp2": "🚨 Categoría CLP 2: Muy Inflamable",
        "clp3": "⚠️ Categoría CLP 3: Inflamable",
        "non_flam": "✅ No clasificado por inflamabilidad",
        "clp_reg": "Reglamento CLP",
        "voc": "Estado de COV (2004/42/CE)",
        "bio_voc": "Biológico/Bajo COV",
        "batch_title": "🏭 Calculadora de Producción por Lotes",
        "batch_vol": "Volumen del lote (Litros)",
        "brine_conc": "Concentración de salmuera de stock (% NaCl)",
        "total_mass": "Masa Total del Lote",
        "total_cost": "Coste Estimado de Materiales",
        "weighing": "📋 Instrucciones de Pesada",
        "comp": "Componente",
        "weight": "Peso (kg)",
        "step": "Paso",
        "brine_stock": "Salmuera de Stock",
        "pure_water": "Agua Pura",
        "brine_err": "Error: La salmuera de stock está demasiado diluida.",
        "tip": "💡 **Consejo de mezcla:** Mezcle primero la fase orgánica (pasos 1-4), luego añada la fase acuosa (paso 5) con agitación persistente.",
        "report_title": "📄 Generación de Informe",
        "tds_btn": "📥 Descargar Ficha Técnica (TXT)",
        "ref_data": "📚 Datos científicos de referencia",
        "origin": "Origen",
        "code": "Código",
        "statement": "Indicación",
        "precaution": "Precaución"
    },
    "Galego": {
        "title": "🧪 Suite de Fabricación de Microemulsións Industriais",
        "designer": "🛠️ Deseñador de Formulación",
        "selection": "Selección de Compoñentes",
        "resin": "Resina Obxectivo",
        "solvent": "Solvente Base",
        "cosolvent": "Cosolvente de Nivelación",
        "surfactant": "Tansioactivo Principal",
        "cosurfactant": "Cotansioactivo (Alcohol)",
        "concs": "Concentracións (%)",
        "p_sol": "% Fase Solvente",
        "p_cos": "% Cosolvente",
        "p_sur": "% Tansioactivo",
        "p_alc": "% Cotansioactivo",
        "wat_err": "Demasiada fase orgánica! A auga debe ser polo menos o 5%.",
        "wat_info": "💧 Fase Acuosa: ",
        "salinity": "Salinidade (% NaCl na mestura)",
        "tabs": ["📊 Análise de Laboratorio", "🛡️ Seguridade e Normativa", "🏭 Fabricación"],
        "hsp_title": "Espazo de Solubilidade de Hansen (3D)",
        "target": "Obxectivo",
        "current": "Formulación Actual",
        "center": "Centro",
        "perf": "Indicadores de Rendemento",
        "red": "RED de Solubilidade",
        "hld": "HLD de Estabilidade",
        "phys": "Perfil Físico",
        "density": "Densidade",
        "fp": "Punto de Inflamación",
        "stability": "Estabilidade",
        "solubility": "Solubilidade",
        "optimal": "Óptimo",
        "unstable": "Inestable",
        "high": "Alta",
        "medium": "Media",
        "low": "Baixa",
        "excellent": "Excelente",
        "partial": "Parcial",
        "poor": "Pobre",
        "opt_btn": "🚀 Optimizador: Minimizar RED e HLD",
        "opt_wait": "Calculando proporcións óptimas...",
        "opt_succ": "Optimizado! Solvent: ",
        "opt_err": "Non converxeu. Probe outro punto de inicio.",
        "reg_title": "Cumprimento Normativo e GHS",
        "h_phrases": "### ⚠️ Indicacións de Perigo (Frases H)",
        "p_phrases": "### 🛡️ Consellos de Prudencia (Frases P)",
        "flam_title": "🔥 Clasificación de Inflamabilidade CLP/ECHA",
        "calc_fp": "Punto de Inflamación calculado",
        "clp2": "🚨 Categoría CLP 2: Moi Inflamable",
        "clp3": "⚠️ Categoría CLP 3: Inflamable",
        "non_flam": "✅ Non clasificado por inflamabilidade",
        "clp_reg": "Regulamento CLP",
        "voc": "Estado de COV (2004/42/CE)",
        "bio_voc": "Biolóxico/Baixo COV",
        "batch_title": "🏭 Calculadora de Produción por Lotes",
        "batch_vol": "Volume do lote (Litros)",
        "brine_conc": "Concentración de salmoira de stock (% NaCl)",
        "total_mass": "Masa Total do Lote",
        "total_cost": "Custo Estimado de Materiais",
        "weighing": "📋 Instrucións de Pesada",
        "comp": "Compoñente",
        "weight": "Peso (kg)",
        "step": "Paso",
        "brine_stock": "Salmoira de Stock",
        "pure_water": "Auga Pura",
        "brine_err": "Erro: A salmoira de stock está demasiado diluída.",
        "tip": "💡 **Consello de mestura:** Mesture primeiro a fase orgánica (pasos 1-4), despois engada a fase acuosa (paso 5) con axitación persistente.",
        "report_title": "📄 Xeración de Informe",
        "tds_btn": "📥 Descargar Ficha Técnica (TXT)",
        "ref_data": "📚 Datos científicos de referencia",
        "origin": "Orixe",
        "code": "Código",
        "statement": "Indicación",
        "precaution": "Precaución"
    }
}

DATA_FULL = {
    # We use English keys for internally matching but localized display where needed
    "Solvents": {
        "Methyl Sunflowerate": {"hsp": [16.2, 3.2, 3.8], "rho": 880, "eacn": 1.5, "fp": 170, "price": 1.65, "ghs": []},
        "Methyl Soyate": {"hsp": [16.1, 3.1, 3.7], "rho": 885, "eacn": 1.4, "fp": 175, "price": 1.60, "ghs": []},
        "Methyl Palmitate": {"hsp": [16.3, 3.3, 3.9], "rho": 870, "eacn": 1.8, "fp": 180, "price": 1.55, "ghs": []},
        "DBE (Dibasic Esters)": {"hsp": [16.5, 7.5, 7.0], "rho": 1060, "eacn": -5.4, "fp": 108, "price": 2.85, "ghs": ["H319", "P264", "P280", "P305+P351+P338"]},
        "Toluene": {"hsp": [18.0, 1.4, 2.0], "rho": 867, "eacn": 1.0, "fp": 4, "price": 0.95, "ghs": ["H225", "H304", "H315", "H336", "H361d", "P210", "P260", "P280", "P301+P310", "P331"]},
        "Butyl Acetate": {"hsp": [15.8, 3.7, 6.3], "rho": 881, "eacn": -2.5, "fp": 22, "price": 1.25, "ghs": ["H226", "H336", "P210", "P261", "P304+P340"]},
    },
    "Cosolvents": {
        "None": {"hsp": [0, 0, 0], "rho": 1.0, "f_hld": 0, "fp": 200, "price": 0, "ghs": []},
        "Benzilic Alcohol": {"hsp": [18.4, 6.3, 13.7], "rho": 1045, "f_hld": -0.2, "fp": 93, "price": 2.45, "ghs": ["H302", "H332", "P261", "P264", "P270"]},
        "Propylene Carbonate": {"hsp": [20.0, 18.0, 4.1], "rho": 1200, "f_hld": 0.1, "fp": 132, "price": 2.10, "ghs": ["H319", "P264", "P280"]},
        "Glycerin": {"hsp": [17.4, 12.1, 29.3], "rho": 1260, "f_hld": -0.8, "fp": 160, "price": 0.95, "ghs": []},
        "Butyl Diglycol": {"hsp": [16.0, 7.0, 10.6], "rho": 953, "f_hld": -0.2, "fp": 105, "price": 3.20, "ghs": ["H319", "P264", "P280"]},
        "Cyrene": {"hsp": [18.9, 12.4, 7.1], "rho": 1250, "f_hld": 0.4, "fp": 108, "price": 8.50, "ghs": ["H319", "P264", "P280"]},
        "Acetone": {"hsp": [15.5, 10.4, 7.0], "rho": 784, "f_hld": 0.0, "fp": -20, "price": 0.85, "ghs": ["H225", "H319", "H336", "P210", "P261", "P305+P351+P338"]},
        "Loxanol (Generic)": {"hsp": [16.5, 4.2, 8.5], "rho": 980, "f_hld": -0.1, "fp": 105, "price": 4.20, "ghs": []},
    },
    "Surfactants": {
        "APG (Non-Ionic)": {"cc": 1.5, "hlb": 13.5, "rho": 1100, "hsp": [18.0, 12.0, 15.0], "price": 4.50, "ghs": ["H318", "P280", "P305+P351+P338"], "type": "NI", "fp": 200},
        "SLES (Anionic)": {"cc": -2.0, "hlb": 40.0, "rho": 1050, "hsp": [17.5, 11.0, 9.5], "price": 3.85, "ghs": ["H315", "H318", "P264", "P280", "P302+P352"], "type": "A", "fp": 200},
        "Rokawin NL7": {"cc": 1.2, "hlb": 12.0, "rho": 1010, "hsp": [16.5, 6.0, 10.0], "price": 3.10, "ghs": ["H302", "H318", "P264", "P280", "P301+P312"], "type": "NI", "fp": 150},
        "Kolliphor P188": {"cc": 3.5, "hlb": 29.0, "rho": 1060, "hsp": [17.0, 8.0, 12.0], "price": 12.50, "ghs": [], "type": "NI", "fp": 250},
        "Tween 80": {"cc": 2.2, "hlb": 15.0, "rho": 1070, "hsp": [17.5, 10.5, 14.0], "price": 5.80, "ghs": [], "type": "NI", "fp": 113},
    },
    "Cosurfactants": {
        "Ethanol": {"hsp": [15.8, 8.8, 19.4], "rho": 789, "fp": 13, "price": 1.20, "ghs": ["H225", "H319", "P210", "P233", "P280"]},
        "Butanol": {"hsp": [16.0, 5.7, 15.8], "rho": 810, "fp": 35, "price": 1.80, "ghs": ["H226", "H302", "H315", "H318", "H335", "P210", "P261", "P280"]},
        "Isopropanol (IPA)": {"hsp": [15.8, 6.1, 16.4], "rho": 786, "fp": 12, "price": 1.50, "ghs": ["H225", "H319", "H336", "P210", "P261", "P280"]},
    },
    "Resins": {
        "Alkydic": {"hsp": [18.5, 4.5, 5.1], "r0": 8.0},
        "Nitrocellulose": {"hsp": [15.4, 10.1, 8.8], "r0": 11.5},
        "Polyurethane": {"hsp": [17.8, 10.5, 11.2], "r0": 9.0},
        "PVC": {"hsp": [18.8, 9.2, 3.5], "r0": 7.5},
        "Acrylic": {"hsp": [18.5, 9.0, 7.5], "r0": 9.2},
        "Chlorinated Rubber": {"hsp": [18.0, 6.0, 7.0], "r0": 8.5},
        "Synthetic": {"hsp": [17.5, 3.5, 3.0], "r0": 7.0},
        "Epoxy-Polyamide": {"hsp": [18.5, 9.5, 10.0], "r0": 11.0},
    }
}

GHS_MAP_GLOBAL = {
    # We define translation inside local logic
}

PICTOGRAMS = {
    "Flame": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/GHS-pictogram-flamme.svg/100px-GHS-pictogram-flamme.svg.png",
    "Exclamation": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/GHS-pictogram-exclam.svg/100px-GHS-pictogram-exclam.svg.png",
    "Health": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/GHS-pictogram-silhouete.svg/100px-GHS-pictogram-silhouete.svg.png",
    "Corrosive": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/GHS-pictogram-acid.svg/100px-GHS-pictogram-acid.svg.png",
    "Toxic": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/GHS-pictogram-skull.svg/100px-GHS-pictogram-skull.svg.png",
    "Environment": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/GHS-pictogram-pollut.svg/100px-GHS-pictogram-pollut.svg.png"
}

# --- 2. LOGIC FUNCTIONS ---
def calculate_hld(sf, oil_eacn, salinity, temp=25):
    k = 0.17
    s = salinity if salinity > 0 else 0.001
    if sf.get("type") == "A": return np.log(s) - k * oil_eacn + sf["cc"]
    else: return 0.13 * s - k * oil_eacn + sf["cc"]

def calculate_red(h_mix, resin):
    ra = np.sqrt(4 * (h_mix[0] - resin["hsp"][0])**2 + (h_mix[1] - resin["hsp"][1])**2 + (h_mix[2] - resin["hsp"][2])**2)
    return ra / resin["r0"]

def get_physical_properties(ws, rhos, fps):
    rho_mix = 1 / np.sum(ws / rhos)
    fp_mix = 10**np.sum(ws * np.log10(fps))
    return rho_mix, fp_mix

# --- 3. UI SETUP ---
st.set_page_config(page_title="MicroSaaS Pro", layout="wide", page_icon="🧪")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f1f5f9; }
    .stMetric { background-color: #1e293b !important; padding: 20px !important; border-radius: 12px; border: 1px solid #334155; }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0f172a; }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom-color: #38bdf8 !important; }
    [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
    .stTable, .stDataFrame { background-color: #1e293b; color: #f1f5f9; border-radius: 8px; }
    div[data-testid="stExpander"] { background-color: #1e293b !important; border: 1px solid #334155 !important; }
    .stAlert { background-color: #1e293b !important; color: #f1f5f9 !important; border: 1px solid #334155 !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3655/3655555.png", width=50)
    lang_choice = st.selectbox("🌐 Language / Idioma / Lingua", ["English", "Español", "Galego"])
    L = LANGS[lang_choice]
    st.markdown("---")
    st.header(L["designer"])
    
    with st.expander(L["selection"], expanded=True):
        res_k = st.selectbox(L["resin"], list(DATA_FULL["Resins"].keys()))
        sol_k = st.selectbox(L["solvent"], list(DATA_FULL["Solvents"].keys()))
        cos_k = st.selectbox(L["cosolvent"], list(DATA_FULL["Cosolvents"].keys()))
        sur_k = st.selectbox(L["surfactant"], list(DATA_FULL["Surfactants"].keys()))
        alc_k = st.selectbox(L["cosurfactant"], list(DATA_FULL["Cosurfactants"].keys()))
    
    with st.expander(L["concs"], expanded=True):
        p_sol = st.slider(L["p_sol"], 5, 60, 25)
        p_cos = st.slider(L["p_cos"], 0, 15, 5)
        p_sur = st.slider(L["p_sur"], 5, 30, 12)
        p_alc = st.slider(L["p_alc"], 2, 20, 8)
        p_wat = 100 - p_sol - p_cos - p_sur - p_alc
        if p_wat < 5: st.error(L["wat_err"]); p_wat = 5
        st.info(f"{L['wat_info']} {p_wat}%")
        salinity = st.number_input(L["salinity"], 0.0, 10.0, 0.5, step=0.1)

# Logic Prep
S, C, Sf, A, R = DATA_FULL["Solvents"][sol_k], DATA_FULL["Cosolvents"][cos_k], DATA_FULL["Surfactants"][sur_k], DATA_FULL["Cosurfactants"][alc_k], DATA_FULL["Resins"][res_k]
W = {"hsp": [15.5, 16.0, 42.1], "rho": 997, "fp": 1000, "price": 0.01}
comps = [S, C, Sf, A, W]
ws = np.array([p_sol, p_cos, p_sur, p_alc, p_wat]) / 100
rho_mix, fp_mix = get_physical_properties(ws, np.array([c["rho"] for c in comps]), np.array([c.get("fp", 200) for c in comps]))
v_fracs = (ws / np.array([c["rho"] for c in comps])) / np.sum(ws / np.array([c["rho"] for c in comps]))
h_mix = np.dot(v_fracs, np.array([c["hsp"] for c in comps]))
red = calculate_red(h_mix, R)
oil_eacn = (p_sol * S["eacn"] + p_cos * C.get("f_hld", 0)) / (p_sol + p_cos) if (p_sol + p_cos) > 0 else 0
hld = calculate_hld(Sf, oil_eacn, salinity)

st.title(L["title"])
st.markdown("---")

t1, t2, t3 = st.tabs(L["tabs"])

with t1:
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.subheader(L["hsp_title"])
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x_s = R['r0'] * np.cos(u) * np.sin(v) + R['hsp'][0]
        y_s = R['r0'] * np.sin(u) * np.sin(v) + R['hsp'][1]
        z_s = R['r0'] * np.cos(v) + R['hsp'][2]
        fig = go.Figure()
        fig.add_trace(go.Surface(x=x_s, y=y_s, z=z_s, opacity=0.4, showscale=False, colorscale=[[0, '#38bdf8'], [1, '#38bdf8']],
            contours=dict(x=dict(show=True, color="white", width=1), y=dict(show=True, color="white", width=1), z=dict(show=True, color="white", width=1)),
            name=f"{L['target']}: {res_k}"))
        fig.add_trace(go.Scatter3d(x=[h_mix[0]], y=[h_mix[1]], z=[h_mix[2]], mode='markers', marker=dict(size=12, color='#ef4444', symbol='diamond', line=dict(color='white', width=2)), name=L["current"]))
        fig.add_trace(go.Scatter3d(x=[R['hsp'][0]], y=[R['hsp'][1]], z=[R['hsp'][2]], mode='markers', marker=dict(size=6, color='#fbbf24'), name=f"{res_k} {L['center']}"))
        ctr, rv = R['hsp'], R['r0'] + 5
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', scene=dict(
            xaxis=dict(title='δD', range=[ctr[0]-rv, ctr[0]+rv], gridcolor="#475569"),
            yaxis=dict(title='δP', range=[ctr[1]-rv, ctr[1]+rv], gridcolor="#475569"),
            zaxis=dict(title='δH', range=[ctr[2]-rv, ctr[2]+rv], gridcolor="#475569"),
            aspectmode='cube', bgcolor="#0f172a"
        ), margin=dict(l=0, r=0, b=0, t=0), legend=dict(x=0.02, y=0.98, font=dict(color="white"), bgcolor="rgba(15, 23, 42, 0.8)"))
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.subheader(L["perf"])
        m1, m2 = st.columns(2)
        m1.metric(L["red"], f"{red:.2f}", delta="-0.05" if red < 1 else "+0.05", delta_color="inverse")
        m2.metric(L["hld"], f"{hld:.2f}", delta=L["optimal"] if abs(hld) < 0.5 else L["unstable"])
        st.markdown(f"### {L['phys']}")
        tds_dict = {L["density"]: f"{rho_mix:.1f} kg/m³", L["fp"]: f"{fp_mix:.1f} °C",
                    L["stability"]: f"✅ {L['high']}" if abs(hld) < 0.5 else f"⚠️ {L['medium']}" if abs(hld) < 1.0 else f"❌ {L['low']}",
                    L["solubility"]: f"✅ {L['excellent']}" if red < 1 else f"⚠️ {L['partial']}" if red < 1.5 else f"❌ {L['poor']}"}
        st.json(tds_dict)
        if st.button(L["opt_btn"]):
            with st.spinner(L["opt_wait"]):
                def obj(x):
                    p_wat_opt = 100 - x[0] - p_cos - x[1] - x[2]
                    if p_wat_opt < 10: return 1e6
                    ws_opt = np.array([x[0], p_cos, x[1], x[2], p_wat_opt]) / 100
                    v_o = (ws_opt / np.array([c["rho"] for c in comps])) / np.sum(ws_opt / np.array([c["rho"] for c in comps]))
                    h_o = np.dot(v_o, np.array([c["hsp"] for c in comps]))
                    r_o = calculate_red(h_o, R)
                    o_e = (x[0] * S["eacn"] + p_cos * C.get("f_hld", 0)) / (x[0] + p_cos)
                    h_o_v = calculate_hld(Sf, o_e, x[3])
                    return 10*h_o_v**2 + 5*r_o**2 + 0.1*x[1]
                res = minimize(obj, [p_sol, p_sur, p_alc, salinity], bounds=[(10, 50), (5, 25), (2, 15), (0, 5)])
                if res.success: st.success(f"{L['opt_succ']} {res.x[0]:.1f}%, Surf: {res.x[1]:.1f}%, Alc: {res.x[2]:.1f}%, Sal: {res.x[3]:.1f}%")
                else: st.warning(L["opt_err"])

with t2:
    st.subheader(L["reg_title"])
    # Simplified GHS logic for unified app
    GHS_TEXTS = {
        "English": {
            "H225": "Highly flammable liquid and vapour", "H226": "Flammable liquid and vapour",
            "H302": "Harmful if swallowed", "H304": "May be fatal if swallowed and enters airways",
            "H315": "Causes skin irritation", "H318": "Causes serious eye damage", "H319": "Causes serious eye irritation",
            "H332": "Harmful if inhaled", "H335": "May cause respiratory irritation", "H336": "May cause drowsiness or dizziness",
            "H361d": "Suspected of damaging the unborn child"
        },
        "Español": {
            "H225": "Líquido y vapores muy inflamables", "H226": "Líquidos y vapores inflamables",
            "H302": "Nocivo en caso de ingestión", "H304": "Dañino en caso de ingestión pulmonar",
            "H315": "Provoca irritación cutánea", "H318": "Provoca lesiones oculares graves", "H319": "Provoca irritación ocular grave",
            "H332": "Nocivo en caso de inhalación", "H335": "Puede irritar las vías respiratorias", "H336": "Somnolencia o vértigo",
            "H361d": "Sospecha de que daña al feto"
        },
        "Galego": {
            "H225": "Líquido e vapores moi inflamables", "H226": "Líquido e vapores inflamables",
            "H302": "Nocivo en caso de inxestión", "H304": "Mortal en caso de inxestión e penetración pulmonar",
            "H315": "Irritación cutánea", "H318": "Lesións oculares graves", "H319": "Irritación ocular grave",
            "H332": "Nocivo por inhalación", "H335": "Irritación respiratoria", "H336": "Somnolencia ou vertixe",
            "H361d": "Risco para o feto"
        }
    }
    
    pictos, h_list, p_list = set(), [], []
    H_KEYS = {"Flame": ["H225", "H226"], "Exclamation": ["H302", "H315", "H319", "H332", "H335", "H336"], "Corrosive": ["H318"], "Health": ["H304", "H361d"]}
    
    for name, c, w in [(sol_k, S, p_sol/100), (cos_k, C, p_cos/100), (sur_k, Sf, p_sur/100), (alc_k, A, p_alc/100)]:
        if w > 0.05:
            for code in c.get("ghs", []):
                if code.startswith("H"):
                    h_list.append({L["code"]: code, L["statement"]: GHS_TEXTS[lang_choice].get(code, code), L["origin"]: name})
                    for k, v in H_KEYS.items():
                        if code in v: pictos.add(k)
                elif code.startswith("P"): p_list.append({L["code"]: code, L["origin"]: name})

    if pictos:
        cols = st.columns(len(pictos))
        for i, pt in enumerate(pictos):
            with cols[i]: st.image(PICTOGRAMS[pt], width=80); st.caption(f"GHS {pt}")
    if h_list: st.markdown(f"### {L['h_phrases']}"); st.table(pd.DataFrame(h_list))
    if p_list: st.markdown(f"### {L['p_phrases']}"); st.table(pd.DataFrame(p_list))

    st.markdown("---")
    st.markdown(f"### {L['flam_title']}")
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**{L['calc_fp']}:** {fp_mix:.1f} °C")
        if fp_mix < 23: st.error(L["clp2"]); st.caption("< 23°C")
        elif 23 <= fp_mix <= 60: st.warning(L["clp3"]); st.caption("23°C - 60°C")
        else: st.success(L["non_flam"]); st.caption("> 60°C")
    with c2:
        voc = "Yes/Si" if S["fp"] < 250 else L["bio_voc"]
        st.write(f"**{L['voc']}:** {voc}")

with t3:
    st.header(L["batch_title"])
    c_b1, c_b2 = st.columns(2)
    with c_b1:
        bv = st.number_input(L["batch_vol"], 100, 50000, 1000)
        bc = st.slider(L["brine_conc"], 5.0, 25.0, 15.0)
    total_kg = bv * (rho_mix / 1000)
    cost = np.sum(ws * np.array([c.get("price", 0) for c in comps])) * total_kg
    with c_b2:
        st.metric(L["total_mass"], f"{total_kg:.1f} kg")
        st.metric(L["total_cost"], f"€{cost:.2f}")

    st.markdown(f"### {L['weighing']}")
    target_s = total_kg * (salinity/100)
    w_b = target_s / (bc/100)
    w_w = (total_kg * (p_wat/100)) - w_b
    if w_w < 0: st.error(L["brine_err"])
    batch_df = pd.DataFrame({
        L["comp"]: [sol_k, cos_k, sur_k, alc_k, L["brine_stock"], L["pure_water"]],
        L["weight"]: [total_kg*(p_sol/100), total_kg*(p_cos/100), total_kg*(p_sur/100), total_kg*(p_alc/100), w_b, w_w],
        L["step"]: [1, 2, 3, 4, 5, 5]
    })
    st.table(batch_df)
    st.success(L["tip"])
    st.markdown("---")
    st.subheader(L["report_title"])
    rep = f"TDS - {sol_k}\nMix: {p_sol}/{p_cos}/{p_sur}/{p_alc}/{p_wat}\nDensity: {rho_mix:.1f}\nFP: {fp_mix:.1f}\nHLD: {hld:.2f}\nRED: {red:.2f}"
    st.download_button(L["tds_btn"], rep, file_name="TDS.txt")

with st.expander(L["ref_data"]):
    for k, v in DATA_FULL.items():
        st.write(f"**{k}**")
        st.dataframe(pd.DataFrame(v).T)
