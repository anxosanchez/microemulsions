import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize

# --- 1. GLOBAL DATABASE ---
DATA = {
    "Solvents (Oil Phase)": {
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
    "Cosurfactants (Alcohols)": {
        "Ethanol": {"hsp": [15.8, 8.8, 19.4], "rho": 789, "fp": 13, "price": 1.20, "ghs": ["H225", "H319", "P210", "P233", "P280"]},
        "Butanol": {"hsp": [16.0, 5.7, 15.8], "rho": 810, "fp": 35, "price": 1.80, "ghs": ["H226", "H302", "H315", "H318", "H335", "P210", "P261", "P280"]},
        "Isopropanol (IPA)": {"hsp": [15.8, 6.1, 16.4], "rho": 786, "fp": 12, "price": 1.50, "ghs": ["H225", "H319", "H336", "P210", "P261", "P280"]},
    },
    "Resins (Targets)": {
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

GHS_MAP = {
    # H-Phrases
    "H225": {"text": "Highly flammable liquid and vapour", "type": "Flame", "cat": "H"},
    "H226": {"text": "Flammable liquid and vapour", "type": "Flame", "cat": "H"},
    "H302": {"text": "Harmful if swallowed", "type": "Exclamation", "cat": "H"},
    "H304": {"text": "May be fatal if swallowed and enters airways", "type": "Health", "cat": "H"},
    "H315": {"text": "Causes skin irritation", "type": "Exclamation", "cat": "H"},
    "H318": {"text": "Causes serious eye damage", "type": "Corrosive", "cat": "H"},
    "H319": {"text": "Causes serious eye irritation", "type": "Exclamation", "cat": "H"},
    "H332": {"text": "Harmful if inhaled", "type": "Exclamation", "cat": "H"},
    "H335": {"text": "May cause respiratory irritation", "type": "Exclamation", "cat": "H"},
    "H336": {"text": "May cause drowsiness or dizziness", "type": "Exclamation", "cat": "H"},
    "H361d": {"text": "Suspected of damaging the unborn child", "type": "Health", "cat": "H"},
    # P-Phrases
    "P210": {"text": "Keep away from heat, hot surfaces, sparks, open flames and other ignition sources. No smoking.", "cat": "P"},
    "P233": {"text": "Keep container tightly closed.", "cat": "P"},
    "P260": {"text": "Do not breathe dust/fume/gas/mist/vapours/spray.", "cat": "P"},
    "P261": {"text": "Avoid breathing dust/fume/gas/mist/vapours/spray.", "cat": "P"},
    "P264": {"text": "Wash hands thoroughly after handling.", "cat": "P"},
    "P270": {"text": "Do not eat, drink or smoke when using this product.", "cat": "P"},
    "P280": {"text": "Wear protective gloves/protective clothing/eye protection/face protection.", "cat": "P"},
    "P301+P310": {"text": "IF SWALLOWED: Immediately call a POISON CENTER/doctor.", "cat": "P"},
    "P301+P312": {"text": "IF SWALLOWED: Call a POISON CENTER/doctor if you feel unwell.", "cat": "P"},
    "P302+P352": {"text": "IF ON SKIN: Wash with plenty of water.", "cat": "P"},
    "P304+P340": {"text": "IF INHALED: Remove person to fresh air and keep comfortable for breathing.", "cat": "P"},
    "P305+P351+P338": {"text": "IF IN EYES: Rinse cautiously with water for several minutes. Remove contact lenses, if present and easy to do. Continue rinsing.", "cat": "P"},
    "P331": {"text": "Do NOT induce vomiting.", "cat": "P"}
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
    """Hydrophilic-Lipophilic Difference Model."""
    k = 0.17
    s = salinity if salinity > 0 else 0.001
    if sf.get("type") == "A":  # Anionic
        return np.log(s) - k * oil_eacn + sf["cc"]
    else:  # Non-Ionic
        b = 0.13
        return b * s - k * oil_eacn + sf["cc"]

def calculate_red(h_mix, resin):
    """Relative Energy Difference for Solubility."""
    ra = np.sqrt(4 * (h_mix[0] - resin["hsp"][0])**2 + 
                 (h_mix[1] - resin["hsp"][1])**2 + 
                 (h_mix[2] - resin["hsp"][2])**2)
    return ra / resin["r0"]

def get_physical_properties(ws, rhos, fps):
    """Estimate mixture properties."""
    # Density (Volume fraction based ideally, but approx here)
    rho_mix = 1 / np.sum(ws / rhos)
    
    # Flash Point (Wickey-Chittenden approx)
    log_fp = np.sum(ws * np.log10(fps))
    fp_mix = 10**log_fp
    
    return rho_mix, fp_mix

# --- 3. UI SETUP ---
st.set_page_config(page_title="MicroSaaS Pro | Lab Dashboard", layout="wide", page_icon="🧪")

# Custom CSS for Premium Dark Laboratory Look
st.markdown("""
<style>
    /* Global Background */
    .stApp { 
        background-color: #0f172a; 
        color: #f1f5f9;
    }
    
    /* Main Content Area */
    .main { 
        background-color: #0f172a;
        padding: 2rem; 
    }

    /* Metric Cards */
    [data-testid="stMetricValue"] { color: #38bdf8 !important; }
    .stMetric { 
        background-color: #1e293b !important; 
        padding: 20px !important; 
        border-radius: 12px !important; 
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3) !important;
        border: 1px solid #334155 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 24px; 
        background-color: #0f172a;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent !important;
        color: #94a3b8 !important;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] { 
        color: #38bdf8 !important; 
        border-bottom-color: #38bdf8 !important; 
    }

    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] .stMarkdown h1, h2, h3 { color: #f1f5f9; }

    /* Headers */
    h1, h2, h3 { color: #f8fafc; font-weight: 800; }
    
    /* JSON/Table/Alerts customization */
    .stTable, .stDataFrame { background-color: #1e293b; color: #f1f5f9; border-radius: 8px; }
    div[data-testid="stExpander"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
    }
    .stAlert {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
    }
    /* Fix for Json background */
    div[data-testid="stJson"] {
        background-color: #1e293b !important;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Industrial Microemulsion Manufacturing Suite")
st.markdown("---")

with st.sidebar:
    st.header("🛠️ Formulation Designer")
    
    with st.expander("Component Selection", expanded=True):
        res_k = st.selectbox("Target Resin", list(DATA["Resins (Targets)"].keys()))
        sol_k = st.selectbox("Base Solvent", list(DATA["Solvents (Oil Phase)"].keys()))
        cos_k = st.selectbox("Leveling Cosolvent", list(DATA["Cosolvents"].keys()))
        sur_k = st.selectbox("Primary Surfactant", list(DATA["Surfactants"].keys()))
        alc_k = st.selectbox("Cosurfactant (Alcohol)", list(DATA["Cosurfactants (Alcohols)"].keys()))
    
    with st.expander("Concentrations (%)", expanded=True):
        p_sol = st.slider("% Solvent Phase", 5, 60, 25)
        p_cos = st.slider("% Cosolvent", 0, 15, 5)
        p_sur = st.slider("% Surfactant", 5, 30, 12)
        p_alc = st.slider("% Cosurfactant", 2, 20, 8)
        p_wat = 100 - p_sol - p_cos - p_sur - p_alc
        
        if p_wat < 5:
            st.error("Too much organic phase! Water must be at least 5%.")
            p_wat = 5
        st.info(f"💧 Aqueous Phase: {p_wat}%")
        
        salinity = st.number_input("Salinity (% NaCl in mix)", 0.0, 10.0, 0.5, step=0.1)

# Calculations Prep
S, C, Sf, A, R = DATA["Solvents (Oil Phase)"][sol_k], DATA["Cosolvents"][cos_k], \
                 DATA["Surfactants"][sur_k], DATA["Cosurfactants (Alcohols)"][alc_k], \
                 DATA["Resins (Targets)"][res_k]

# Water HSP and Props
W = {"hsp": [15.5, 16.0, 42.1], "rho": 997, "fp": 1000, "price": 0.01} # Use high FP for water

components = [S, C, Sf, A, W]
ws = np.array([p_sol, p_cos, p_sur, p_alc, p_wat]) / 100
rhos = np.array([c["rho"] for c in components])
fps = np.array([c.get("fp", 200) for c in components])
hsps = np.array([c["hsp"] for c in components])

rho_mix, fp_mix = get_physical_properties(ws, rhos, fps)

# Volume fractions for HSP (HSP is volume-weighted)
v_fractions = (ws / rhos) / np.sum(ws / rhos)
h_mix = np.dot(v_fractions, hsps)

red = calculate_red(h_mix, R)
oil_eacn = (p_sol * S["eacn"] + p_cos * C.get("f_hld", 0)) / (p_sol + p_cos) if (p_sol + p_cos) > 0 else 0
hld = calculate_hld(Sf, oil_eacn, salinity)

# --- 4. DASHBOARD TABS ---
t1, t2, t3 = st.tabs(["📊 Lab Analysis", "🛡️ Regulatory & Safety", "🏭 Manufacturing"])

with t1:
    col_l, col_r = st.columns([3, 2])
    
    with col_l:
        st.subheader("Hansen Solubility Space (3D)")
        
        # Resin Sphere Data
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        x_s = R['r0'] * np.cos(u) * np.sin(v) + R['hsp'][0]
        y_s = R['r0'] * np.sin(u) * np.sin(v) + R['hsp'][1]
        z_s = R['r0'] * np.cos(v) + R['hsp'][2]
        
        fig = go.Figure()
        
        # Wireframe logic: Increase density for better look
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x_s = R['r0'] * np.cos(u) * np.sin(v) + R['hsp'][0]
        y_s = R['r0'] * np.sin(u) * np.sin(v) + R['hsp'][1]
        z_s = R['r0'] * np.cos(v) + R['hsp'][2]

        # Add Wireframe Sphere
        fig.add_trace(go.Surface(
            x=x_s, y=y_s, z=z_s, 
            opacity=0.4, 
            showscale=False,
            colorscale=[[0, '#38bdf8'], [1, '#38bdf8']],
            contours=dict(
                x=dict(show=True, color="white", width=1),
                y=dict(show=True, color="white", width=1),
                z=dict(show=True, color="white", width=1),
            ),
            name=f"Target: {res_k}"
        ))
        
        # Add Mixture Point
        fig.add_trace(go.Scatter3d(x=[h_mix[0]], y=[h_mix[1]], z=[h_mix[2]], 
                                   mode='markers', marker=dict(size=12, color='#ef4444', symbol='diamond', line=dict(color='white', width=2)), 
                                   name="Current Formulation"))
        
        # Add Resin Center
        fig.add_trace(go.Scatter3d(x=[R['hsp'][0]], y=[R['hsp'][1]], z=[R['hsp'][2]], 
                                   mode='markers', marker=dict(size=6, color='#fbbf24'), 
                                   name=f"{res_k} Center"))
        
        # Calculate bounds to maintain spherical shape
        center = R['hsp']
        r = R['r0'] + 5 # Add padding
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis=dict(title='δD', range=[center[0]-r, center[0]+r], gridcolor="#475569", zerolinecolor="#475569"),
                yaxis=dict(title='δP', range=[center[1]-r, center[1]+r], gridcolor="#475569", zerolinecolor="#475569"),
                zaxis=dict(title='δH', range=[center[2]-r, center[2]+r], gridcolor="#475569", zerolinecolor="#475569"),
                aspectmode='cube',
                bgcolor="#0f172a"
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(x=0.02, y=0.98, font=dict(color="white"), bgcolor="rgba(15, 23, 42, 0.8)")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Performance Indicators")
        
        m1, m2 = st.columns(2)
        m1.metric("Solubility RED", f"{red:.2f}", delta="-0.05" if red < 1 else "+0.05", delta_color="inverse")
        m2.metric("Stability HLD", f"{hld:.2f}", delta="Optimal" if abs(hld) < 0.5 else "Unstable")
        
        st.markdown("### Physical Profile")
        tds_data = {
            "Density": f"{rho_mix:.1f} kg/m³",
            "Flash Point": f"{fp_mix:.1f} °C",
            "Stability": "✅ High" if abs(hld) < 0.5 else "⚠️ Medium" if abs(hld) < 1.0 else "❌ Low",
            "Solubility": "✅ Excellent" if red < 1 else "⚠️ Partial" if red < 1.5 else "❌ Poor"
        }
        st.json(tds_data)
        
        if st.button("🚀 Optimizer: Minimize RED & HLD"):
            with st.spinner("Calculating optimal ratios..."):
                def obj(x):
                    # x = [p_sol, p_sur, p_alc, salinity]
                    p_wat_opt = 100 - x[0] - p_cos - x[1] - x[2]
                    if p_wat_opt < 10: return 1e6
                    ws_opt = np.array([x[0], p_cos, x[1], x[2], p_wat_opt]) / 100
                    v_opt = (ws_opt / rhos) / np.sum(ws_opt / rhos)
                    h_opt = np.dot(v_opt, hsps)
                    r_opt = calculate_red(h_opt, R)
                    o_eacn = (x[0] * S["eacn"] + p_cos * C.get("f_hld", 0)) / (x[0] + p_cos)
                    h_opt_val = calculate_hld(Sf, o_eacn, x[3])
                    return 10*h_opt_val**2 + 5*r_opt**2 + 0.1*x[1] # Minimize hld^2, red^2, and surfactant cost
                
                res = minimize(obj, [p_sol, p_sur, p_alc, salinity], bounds=[(10, 50), (5, 25), (2, 15), (0, 5)])
                if res.success:
                    st.success(f"Optimized! Set Oil: {res.x[0]:.1f}%, Surf: {res.x[1]:.1f}%, Alcohol: {res.x[2]:.1f}%, Salinity: {res.x[3]:.1f}%")
                else:
                    st.warning("Could not converge. Try different baseline.")

with t2:
    st.subheader("Regulatory Compliance & GHS")
    
    hazardous_comps = []
    global_ghs = set()
    
    # Check all components for Hazards
    check_list = [
        (sol_k, S, p_sol/100),
        (cos_k, C, p_cos/100),
        (sur_k, Sf, p_sur/100),
        (alc_k, A, p_alc/100)
    ]
    
    relevant_h = []
    relevant_p = []
    pictos_to_show = set()

    for name, comp, weight in check_list:
        if weight > 0.05: # GHS threshold 5%
            for code in comp.get("ghs", []):
                if code in GHS_MAP:
                    data = GHS_MAP[code]
                    if data["cat"] == "H":
                        relevant_h.append({"Code": code, "Statement": data["text"], "Origin": name})
                        if "type" in data:
                            pictos_to_show.add(data["type"])
                    else:
                        relevant_p.append({"Code": code, "Precaution": data["text"], "Origin": name})

    # Display Pictograms
    if pictos_to_show:
        pic_cols = st.columns(len(pictos_to_show))
        for i, p_type in enumerate(pictos_to_show):
            with pic_cols[i]:
                st.image(PICTOGRAMS[p_type], width=80)
                st.caption(f"GHS {p_type}")

    # H-Phrases
    if relevant_h:
        st.markdown("### ⚠️ Hazard Statements (H-Phrases)")
        st.table(pd.DataFrame(relevant_h))
    
    # P-Phrases
    if relevant_p:
        st.markdown("### �️ Precautionary Statements (P-Phrases)")
        st.table(pd.DataFrame(relevant_p))

    st.markdown("---")
    st.markdown("### 🔥 CLP/ECHA Flammability Classification")
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**Calculated FP:** {fp_mix:.1f} °C")
        # ECHA/CLP Categories for liquids
        if fp_mix < 23:
            st.error("🚨 CLP Category 2: Highly Flammable")
            st.caption("Flash point < 23°C")
        elif 23 <= fp_mix <= 60:
            st.warning("⚠️ CLP Category 3: Flammable")
            st.caption("23°C ≤ Flash point ≤ 60°C")
        else:
            st.success("✅ Not Classified for Flammability")
            st.caption("Flash point > 60°C (CLP Regulation)")
    with c2:
        voc = "Yes" if S["fp"] < 250 else "Biological/Low VOC"
        st.write(f"**VOC Status (2004/42/EC):** {voc}")

with t3:
    st.header("🏭 Batch Production Calculator")
    
    col_batch1, col_batch2 = st.columns(2)
    with col_batch1:
        batch_vol = st.number_input("Batch Volume (Liters)", 100, 50000, 1000)
        brine_conc = st.slider("Stock Brine Concentration (% NaCl)", 5.0, 25.0, 15.0)
    
    total_kg = batch_vol * (rho_mix / 1000)
    
    # Costs
    cost_total = np.sum(ws * np.array([c.get("price", 0) for c in components])) * total_kg
    
    with col_batch2:
        st.metric("Total Batch Mass", f"{total_kg:.1f} kg")
        st.metric("Estimated Material Cost", f"€{cost_total:.2f}")

    st.markdown("### 📋 Weighing Instructions")
    
    # Detailed weights
    w_sol = total_kg * (p_sol/100)
    w_cos = total_kg * (p_cos/100)
    w_sur = total_kg * (p_sur/100)
    w_alc = total_kg * (p_alc/100)
    
    target_salt = total_kg * (salinity / 100)
    w_brine = target_salt / (brine_conc / 100)
    w_water = (total_kg * (p_wat / 100)) - w_brine
    
    if w_water < 0:
        st.error("Error: Brine stock is too diluted to reach target salinity with this water fraction.")
    
    production_df = pd.DataFrame({
        "Component": [sol_k, cos_k, sur_k, alc_k, "Brine Stock", "Pure Water"],
        "Weight (kg)": [w_sol, w_cos, w_sur, w_alc, w_brine, w_water],
        "Step": [1, 2, 3, 4, 5, 5]
    })
    st.table(production_df)
    
    st.success("💡 **Mixing Tip:** Blend organic phase (Steps 1-4) first, then slowly add aqueous phase (Step 5) under high shear.")

    st.markdown("---")
    # Export Section
    st.subheader("📄 Report Generation")
    
    report_text = f"""TECHNICAL DATA SHEET
Project: Microemulsion Batch - {sol_k}
--------------------------------------
Physical Properties:
- Density: {rho_mix:.1f} kg/m3
- Flash Point: {fp_mix:.1f} C
- Stability (HLD): {hld:.2f}
- Solubility (RED vs {res_k}): {red:.2f}

Formulation:
- {sol_k}: {p_sol}%
- {cos_k}: {p_cos}%
- {sur_k}: {p_sur}%
- {alc_k}: {p_alc}%
- Water: {p_wat}%
--------------------------------------
GHS Hazards: {", ".join(global_ghs) if global_ghs else "None"}
"""
    st.download_button("📥 Download Technical Data Sheet (TXT)", report_text, file_name="TDS_Microemulsion.txt")

# --- 5. REFERENCE DATA ---
with st.expander("📚 Scientific Reference Data"):
    st.markdown("Current database parameters used for this simulation:")
    for cat, items in DATA.items():
        st.write(f"**{cat}**")
        st.dataframe(pd.DataFrame(items).T)

