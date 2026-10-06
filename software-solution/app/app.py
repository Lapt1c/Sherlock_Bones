"""Aplicație demo (etapa 1): vizualizare CBCT + rezultat hard-codat.
Rulare: streamlit run app.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from demo_data import CASES, CLASSES, N, lesion_volume_cm3, make_case

INK, TEAL, VIEW_BG, AMBER = "#12262E", "#0E7C86", "#0B0F12", "#F2A93B"

st.set_page_config(page_title="Suport diagnostic CBCT", page_icon="🦷", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Figtree', system-ui, sans-serif; color: {INK}; }}
.stApp {{ background: #F4F7F8; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid #DDE5E8; }}
.hero h1 {{ font-size: 1.9rem; font-weight: 700; margin: 0 0 .2rem 0; letter-spacing: -0.01em; }}
.hero p {{ color: #5B7078; margin: 0 0 1.2rem 0; font-size: 1.02rem; }}
.card {{ background: #FFFFFF; border: 1px solid #DDE5E8; border-radius: 14px; padding: 1.1rem 1.2rem; }}
.result-label {{ color: #5B7078; font-size: .85rem; margin-bottom: .15rem; }}
.result-class {{ font-size: 1.55rem; font-weight: 700; line-height: 1.2; margin-bottom: .5rem; }}
.pill {{ display: inline-block; padding: .22rem .7rem; border-radius: 999px; font-size: .85rem; font-weight: 600;
        background: #E3F3F4; color: #0A5A62; }}
.pill.warn {{ background: #FDEFD6; color: #8A5A00; }}
.bar-row {{ margin: .55rem 0; }}
.bar-top {{ display: flex; justify-content: space-between; font-size: .92rem; margin-bottom: .2rem; }}
.bar-bg {{ background: #E8EEF0; border-radius: 6px; height: 9px; overflow: hidden; }}
.bar-fg {{ height: 100%; border-radius: 6px; background: #A9BDC3; }}
.bar-fg.top {{ background: {TEAL}; }}
.stat {{ display: flex; justify-content: space-between; padding: .5rem 0; border-top: 1px solid #EDF2F4; font-size: .95rem; }}
.stat span:first-child {{ color: #5B7078; }}
.note {{ font-size: .85rem; color: #5B7078; margin-top: .8rem; }}
button[data-baseweb="tab"] {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# ---------- bară laterală ----------
with st.sidebar:
    st.markdown("### Caz de analizat")
    uploaded = st.file_uploader("Încarcă un volum CBCT", type=["nrrd", "nii", "gz"],
                                help="Formate acceptate: .nrrd, .nii, .nii.gz")
    case_name = st.selectbox("sau alege un caz demonstrativ", CLASSES)
    if uploaded:
        st.info("Încărcarea reală se conectează în etapa 5. Pentru moment afișăm cazul demonstrativ.")
    st.markdown("### Afișare")
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
    show_contour = st.toggle("Conturul leziunii", value=True)
    show_heat = st.toggle("Zonele care au influențat decizia", value=False)

vol, mask, heat, (cx, cy, cz) = make_case(case_name)
info = CASES[case_name]
probs = info["probs"]
top = int(np.argmax(probs))

# ---------- antet ----------
st.markdown("""
<div class="hero">
  <h1>Analiza leziunilor osoase maxilo-faciale</h1>
  <p>Vizualizează volumul CBCT, vezi unde se află leziunea și ce tip este cel mai probabil.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([2.1, 1], gap="large")


def pane(img, m, h, cross, title):
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=130)
    fig.patch.set_facecolor(VIEW_BG)
    ax.imshow(np.clip(img.T ** (1 / contrast), 0, 1), cmap="gray", origin="lower", vmin=0, vmax=1)
    if show_heat:
        ax.imshow(np.ma.masked_less(h.T, 0.15), cmap="magma", alpha=0.45, origin="lower", vmin=0, vmax=1)
    if show_contour and m.any():
        ax.contour(m.T.astype(float), levels=[0.5], colors=[AMBER], linewidths=1.4, origin="lower")
    ax.axvline(cross[0], color="#FFFFFF", alpha=0.18, lw=0.7)
    ax.axhline(cross[1], color="#FFFFFF", alpha=0.18, lw=0.7)
    ax.set_title(title, color="#C9D6DA", fontsize=9, pad=6)
    ax.axis("off")
    fig.tight_layout(pad=0.6)
    return fig


with left:
    tab_view, tab_why, tab_about = st.tabs(["Vizualizare", "Cum s-a ajuns la rezultat", "Despre model"])

    with tab_view:
        c1, c2, c3 = st.columns(3)
        ax_i = c1.slider("Axial", 0, N - 1, cz, key="ax")
        co_i = c2.slider("Coronal", 0, N - 1, cy, key="co")
        sa_i = c3.slider("Sagital", 0, N - 1, cx, key="sa")
        p1, p2, p3 = st.columns(3)
        figs = [
            pane(vol[:, :, ax_i], mask[:, :, ax_i], heat[:, :, ax_i], (sa_i, co_i), "Axial"),
            pane(vol[:, co_i, :], mask[:, co_i, :], heat[:, co_i, :], (sa_i, ax_i), "Coronal"),
            pane(vol[sa_i, :, :], mask[sa_i, :, :], heat[sa_i, :, :], (co_i, ax_i), "Sagital"),
        ]
        for col, f in zip((p1, p2, p3), figs):
            col.pyplot(f, use_container_width=True)
            plt.close(f)
        st.caption("Linia portocalie încadrează leziunea detectată. Volumul afișat este sintetic (demonstrație).")

    with tab_why:
        st.markdown("**Zonele care au influențat decizia**")
        st.write("Hărțile de explicație arată ce regiuni din volum au contribuit cel mai mult la rezultat, "
                 "ca medicul să poată verifica dacă modelul s-a uitat la locul potrivit. "
                 "Activează „Zonele care au influențat decizia” din bara laterală pentru a le vedea în „Vizualizare”.")
        st.info("Hărțile reale (Grad-CAM 3D) apar după antrenarea modelului. Cele de acum sunt ilustrative.")

    with tab_about:
        st.write("Modelul analizează volumul CBCT în trei pași: găsește leziunea, o conturează și "
                 "o încadrează într-unul din cele patru tipuri din setul de date DOLCHID.")
        st.write("În această versiune, rezultatele sunt fixe și servesc la testarea aplicației.")

with right:
    pills = "warn" if "agresivă" in info["group"] else ""
    bars = "".join(
        f'<div class="bar-row"><div class="bar-top"><span>{n}</span><span>{p*100:.0f}%</span></div>'
        f'<div class="bar-bg"><div class="bar-fg {"top" if i == top else ""}" style="width:{p*100:.0f}%"></div></div></div>'
        for i, (n, p) in enumerate(zip(CLASSES, probs))
    )
    st.markdown(f"""
    <div class="card">
      <div class="result-label">Cel mai probabil</div>
      <div class="result-class">{CLASSES[top]}</div>
      <span class="pill {pills}">{info["group"]}</span>
      <div style="margin-top:1rem">{bars}</div>
      <div style="margin-top:.8rem">
        <div class="stat"><span>Volumul leziunii</span><span><b>{lesion_volume_cm3(mask):.1f} cm³</b></span></div>
        <div class="stat"><span>Încredere</span><span><b>{probs[top]*100:.0f}%</b></span></div>
      </div>
      <div class="note">Instrument de suport pentru decizie. Rezultatul trebuie confirmat de medic și de examenul histopatologic.</div>
    </div>
    """, unsafe_allow_html=True)