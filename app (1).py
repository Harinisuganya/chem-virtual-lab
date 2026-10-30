import re
import html
from io import BytesIO
from urllib.parse import quote

import matplotlib.pyplot as plt
import pandas as pd
import py3Dmol
import requests
import streamlit as st
import streamlit.components.v1 as components
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# -----------------------------------------------------------------------------
# SRM CHEMINFORMATICS VIRTUAL LABORATORY
# Uses PubChem PUG REST for chemistry data; RDKit is intentionally not required.
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="SRM Cheminformatics Virtual Laboratory",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
HEADERS = {"User-Agent": "SRM-Cheminformatics-Virtual-Lab/2.0 (educational Streamlit app)"}
STUDENTS = {
    "Harini M": {"year": "First Year", "section": "Cyber A", "reg": "RA2611030050013"},
    "Harini Sri": {"year": "First Year", "section": "Cyber A", "reg": "RA2611030050010"},
}
EXAMPLES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
}

st.markdown("""
<style>
:root { --srm-blue: #17468f; --srm-gold: #c9a400; }
.block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; }
.hero { background: linear-gradient(120deg,#123c82,#245ca8); color: white;
        padding: 1.35rem 1.5rem; border-radius: 16px; margin-bottom: 1rem; }
.hero h1 { color: white; margin: 0; font-size: clamp(1.6rem,3vw,2.5rem); }
.hero p { color: #edf3ff; margin: .45rem 0 0 0; }
.student-card { border: 1px solid #d8dfeb; border-left: 5px solid #c9a400;
  border-radius: 10px; padding: 12px 14px; margin: 6px 0; }
.small-note { color: #64748b; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)


def api_get(path, params=None, timeout=35):
    """GET a PubChem PUG REST resource and return the response."""
    url = f"{PUBCHEM}/{path.lstrip('/')}"
    response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    return response


@st.cache_data(ttl=3600, show_spinner=False)
def resolve_smiles(smiles):
    """Resolve a SMILES string to a PubChem CID."""
    encoded = quote(smiles.strip(), safe="")
    response = api_get(f"compound/smiles/{encoded}/cids/JSON")
    if response.status_code != 200:
        return None
    data = response.json()
    cids = data.get("IdentifierList", {}).get("CID", [])
    return int(cids[0]) if cids else None


@st.cache_data(ttl=3600, show_spinner=False)
def get_properties(cid):
    """Fetch descriptors from PubChem; unavailable descriptors remain blank."""
    wanted = [
        "MolecularFormula", "MolecularWeight", "ExactMass", "XLogP", "TPSA",
        "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount",
        "HeavyAtomCount", "IUPACName", "ConnectivitySMILES", "SMILES", "InChIKey"
    ]
    fields = ",".join(wanted)
    response = api_get(f"compound/cid/{cid}/property/{fields}/JSON")
    if response.status_code != 200:
        # Retry a smaller set if PubChem does not return one optional descriptor.
        fields = "MolecularFormula,MolecularWeight,ExactMass,XLogP,HeavyAtomCount,IUPACName,ConnectivitySMILES,SMILES,InChIKey"
        response = api_get(f"compound/cid/{cid}/property/{fields}/JSON")
    if response.status_code != 200:
        return {}
    props = response.json().get("PropertyTable", {}).get("Properties", [])
    return props[0] if props else {}


@st.cache_data(ttl=3600, show_spinner=False)
def get_structure_image(cid):
    response = api_get(f"compound/cid/{cid}/PNG", {"image_size": "large"})
    return response.content if response.status_code == 200 and response.headers.get("content-type", "").startswith("image/") else None


@st.cache_data(ttl=3600, show_spinner=False)
def get_sdf(cid, record_type="3d"):
    response = api_get(f"compound/cid/{cid}/SDF", {"record_type": record_type})
    if response.status_code == 200 and "$$$$" in response.text:
        return response.text
    return None


def sdf_bond_summary(sdf_text, formula):
    """Read basic atom/bond counts from an SDF V2000 counts and bond block."""
    lines = sdf_text.splitlines()
    if len(lines) < 5:
        return {}
    try:
        atom_count = int(lines[3][0:3])
        bond_count = int(lines[3][3:6])
        bond_lines = lines[4:4 + bond_count]
        double_count = triple_count = aromatic_count = 0
        for line in bond_lines:
            if len(line) < 9:
                continue
            bond_type = int(line[6:9])
            if bond_type == 2:
                double_count += 1
            elif bond_type == 3:
                triple_count += 1
            elif bond_type == 4:
                aromatic_count += 1
        hydrogen_count = 0
        match = re.search(r"H(\d*)", formula or "")
        if match:
            hydrogen_count = int(match.group(1) or "1")
        return {
            "Number of atoms (SDF record)": atom_count,
            "Heavy-atom bonds": bond_count,
            "Sigma bonds (estimated)": bond_count + hydrogen_count,
            "Pi bonds (estimated)": double_count + (2 * triple_count) + (0.5 * aromatic_count),
            "Hydrogen atoms from formula": hydrogen_count,
        }
    except (ValueError, IndexError):
        return {}


def analyze_smiles(smiles):
    """Resolve and collect data. Returns a record or raises a readable error."""
    cid = resolve_smiles(smiles)
    if cid is None:
        raise ValueError("PubChem could not find an exact matching compound for this SMILES. Check the notation or try a known example.")
    props = get_properties(cid)
    if not props:
        raise ValueError("The compound was found, but PubChem did not return its property record. Please try again later.")
    sdf = get_sdf(cid, "2d")
    bond_info = sdf_bond_summary(sdf, props.get("MolecularFormula", "")) if sdf else {}
    return {"input_smiles": smiles.strip(), "cid": cid, "properties": props, "sdf_2d": sdf, "bond_info": bond_info}


def display_3d(cid, style_name, show_surface):
    sdf = get_sdf(cid, "3d")
    if not sdf:
        st.info("A 3D conformer is not available for this PubChem record. The 2D structure and descriptors are still available.")
        return
    view = py3Dmol.view(width=900, height=560)
    view.addModel(sdf, "sdf")
    styles = {
        "Ball and stick": {"stick": {"radius": 0.16}, "sphere": {"scale": 0.28}},
        "Stick": {"stick": {"radius": 0.20}},
        "Space filling": {"sphere": {"scale": 0.9}},
        "Wireframe": {"line": {"linewidth": 2}},
    }
    view.setStyle(styles[style_name])
    if show_surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.55})
    view.setBackgroundColor("white")
    view.zoomTo()
    components.html(view._make_html(), height=590, scrolling=False)


def safe_value(props, key, suffix=""):
    value = props.get(key)
    if value is None or value == "":
        return "Not available in PubChem"
    if isinstance(value, float):
        value = round(value, 4)
    return f"{value}{suffix}"


def property_rows(record):
    p = record["properties"]
    mapping = [
        ("PubChem CID", record["cid"]),
        ("Molecular formula", p.get("MolecularFormula")),
        ("Molecular weight (g/mol)", p.get("MolecularWeight")),
        ("Exact mass (Da)", p.get("ExactMass")),
        ("XLogP", p.get("XLogP")),
        ("TPSA (Å²)", p.get("TPSA")),
        ("Hydrogen bond donors", p.get("HBondDonorCount")),
        ("Hydrogen bond acceptors", p.get("HBondAcceptorCount")),
        ("Rotatable bonds", p.get("RotatableBondCount")),
        ("Heavy atoms", p.get("HeavyAtomCount")),
        ("IUPAC name", p.get("IUPACName")),
        ("Canonical/connectivity SMILES", p.get("ConnectivitySMILES") or p.get("SMILES")),
        ("InChIKey", p.get("InChIKey")),
    ]
    mapping.extend(record.get("bond_info", {}).items())
    return [(str(k), str(v) if v not in (None, "") else "Not available in PubChem") for k, v in mapping]


def make_pdf(record, student_name, reg_no):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=38, leftMargin=38, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("LabTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#17468f"), fontSize=18, leading=22)
    subtitle = ParagraphStyle("LabSub", parent=styles["Normal"], alignment=TA_CENTER, fontSize=9, textColor=colors.HexColor("#475569"))
    story = [Paragraph("SRM CHEMINFORMATICS VIRTUAL LABORATORY", title),
             Paragraph("Molecular Analysis Report", subtitle), Spacer(1, 18)]
    details = [["Student", student_name], ["Registration number", reg_no],
               ["Input SMILES", record["input_smiles"]], ["PubChem CID", str(record["cid"])]]
    t = Table(details, colWidths=[145, 365])
    t.setStyle(TableStyle([("GRID", (0,0), (-1,-1), .4, colors.grey),
                           ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eaf0f8")),
                           ("VALIGN", (0,0), (-1,-1), "TOP"), ("PADDING", (0,0), (-1,-1), 7)]))
    story += [t, Spacer(1, 16), Paragraph("Molecular properties", styles["Heading2"])]
    data = [["Property", "Value"]] + [[k, v] for k, v in property_rows(record)]
    table = Table(data, colWidths=[190, 320], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17468f")),
                               ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                               ("GRID", (0,0), (-1,-1), .35, colors.grey),
                               ("VALIGN", (0,0), (-1,-1), "TOP"), ("PADDING", (0,0), (-1,-1), 5)]))
    story += [table, Spacer(1, 12), Paragraph("Data source and limitation", styles["Heading2"]),
              Paragraph("Chemical identifiers, descriptors and structure records are retrieved from PubChem PUG REST. Some descriptors may be unavailable for a particular record. Bond counts are estimates parsed from the returned SDF record and molecular formula. This report is intended for educational use.", styles["BodyText"]),
              Spacer(1, 12), Paragraph("Project contributors", styles["Heading2"]),
              Paragraph("Harini M — First Year, Cyber A — RA2611030050013<br/>Harini Sri — First Year, Cyber A — RA2611030050010", styles["BodyText"])]
    doc.build(story)
    return buffer.getvalue()


# Sidebar and student profile
st.sidebar.title("🧪 Virtual Laboratory")
page = st.sidebar.radio("Navigation", [
    "🏠 Home", "📖 Theory", "🧬 Molecular Explorer", "📊 Property Report",
    "📈 Molecule Comparison", "📝 Assessment", "📚 Learning Resources"
])
st.sidebar.divider()
st.sidebar.subheader("👩‍🎓 Student profile")
selected_student = st.sidebar.selectbox("Choose student", list(STUDENTS.keys()))
st.sidebar.caption(f"{STUDENTS[selected_student]['year']} · {STUDENTS[selected_student]['section']}")
st.sidebar.caption(f"Reg. No: {STUDENTS[selected_student]['reg']}")
st.sidebar.divider()
st.sidebar.caption("Chemistry • SMILES • 2D/3D structures • Molecular properties")

# Home
if page == "🏠 Home":
    logo_path = "srm_logo.jpeg"
    try:
        st.image(logo_path, use_container_width=True)
    except Exception:
        st.warning("SRM logo file was not found. Keep srm_logo.jpeg in the same folder as app.py.")
    st.markdown('<div class="hero"><h1>🧪 Cheminformatics Virtual Laboratory</h1><p>Explore molecular structures, descriptors and structure–property relationships.</p></div>', unsafe_allow_html=True)
    st.markdown("### SRM Institute of Science and Technology, Tiruchirappalli")
    st.write("**Faculty of Engineering and Technology · School of Sciences · Division of Chemistry**")
    st.write("This educational virtual laboratory supports the study of molecular representation, structure visualization and computational descriptors using PubChem data.")
    st.markdown("### Student project team")
    c1, c2 = st.columns(2)
    for col, (name, info) in zip((c1, c2), STUDENTS.items()):
        with col:
            st.markdown(f'''<div class="student-card"><b>{html.escape(name)}</b><br>{info['year']} · {info['section']}<br>Registration No: {info['reg']}</div>''', unsafe_allow_html=True)
    st.markdown("### What you can do")
    features = ["Look up a molecule using its SMILES", "View 2D structures and available 3D conformers", "Explore molecular formula, mass, LogP, TPSA and hydrogen-bond descriptors", "Compare several compounds", "Complete a scored chemistry quiz", "Export property tables as CSV and PDF"]
    for item in features:
        st.markdown(f"- ✅ {item}")
    st.markdown("### Try an example")
    st.dataframe(pd.DataFrame([{"Molecule": k, "SMILES": v} for k, v in EXAMPLES.items()]), use_container_width=True, hide_index=True)
    st.caption("Chemical data source: PubChem PUG REST. Some fields depend on PubChem record availability.")

# Theory
elif page == "📖 Theory":
    st.title("📖 Cheminformatics – Theory")
    st.markdown("""
### 1. What is cheminformatics?
Cheminformatics combines chemistry, computer science and data analysis to store, search, represent and study chemical information.

### 2. SMILES notation
**SMILES** means *Simplified Molecular Input Line Entry System*. It represents molecular connectivity as a text string. For example, `CCO` represents ethanol and `c1ccccc1` represents benzene.

### 3. Molecular descriptors
Molecular descriptors are numerical or categorical values that summarize aspects of a molecular structure.

| Descriptor | Meaning |
|---|---|
| Molecular formula | Elemental composition |
| Molecular weight | Average mass of a molecule (g/mol) |
| XLogP | Computed estimate related to octanol/water partitioning |
| TPSA | Topological polar surface area |
| H-bond donors | Count of donor sites |
| H-bond acceptors | Count of acceptor sites |
| Rotatable bonds | Estimate of molecular flexibility |

### 4. Molecular visualization
- **2D structure:** shows atoms and their connectivity in a plane.
- **3D structure:** shows a spatial conformer when one is available in the database.
- **Ball-and-stick:** emphasizes atoms and bonds.
- **Space filling:** emphasizes approximate atomic volume.

### 5. Structure–property relationships
Changes in functional groups, polarity, molecular size and flexibility can influence molecular properties. Comparing related structures helps students explore these relationships.

### 6. Data note
This application retrieves existing PubChem records. It does not independently predict descriptors for compounds absent from PubChem. Always check the source and availability of each field.
""")

# Molecular explorer
elif page == "🧬 Molecular Explorer":
    st.title("🧬 Molecular Explorer")
    st.write("Enter a SMILES string that has a matching PubChem compound record.")
    example_choice = st.selectbox("Load an example", ["Choose an example"] + list(EXAMPLES.keys()))
    default_smiles = EXAMPLES.get(example_choice, "")
    smiles = st.text_input("SMILES notation", value=default_smiles, placeholder="Example: CCO", key="explorer_smiles")
    if st.button("🔎 Analyze molecule", type="primary", key="explorer_analyze"):
        if not smiles.strip():
            st.warning("Enter a SMILES string first.")
        else:
            with st.spinner("Looking up the compound in PubChem..."):
                try:
                    st.session_state["active_record"] = analyze_smiles(smiles)
                    st.success("Compound record retrieved.")
                except requests.RequestException as exc:
                    st.error(f"Network error while contacting PubChem: {exc}")
                except ValueError as exc:
                    st.error(str(exc))
    record = st.session_state.get("active_record")
    if record:
        p = record["properties"]
        st.divider()
        st.subheader(p.get("IUPACName") or "Molecular record")
        st.caption(f"PubChem CID: {record['cid']} · Input SMILES: {record['input_smiles']}")
        left, right = st.columns([1, 1.25])
        with left:
            image_data = get_structure_image(record["cid"])
            if image_data:
                st.image(image_data, caption="PubChem 2D molecular structure", use_container_width=True)
            else:
                st.info("A 2D image is not available for this record.")
        with right:
            st.markdown("#### Key properties")
            metric_cols = st.columns(2)
            metric_cols[0].metric("Molecular formula", p.get("MolecularFormula", "—"))
            metric_cols[1].metric("Molecular weight", f"{p.get('MolecularWeight', '—')} g/mol")
            metric_cols[0].metric("XLogP", p.get("XLogP", "—"))
            metric_cols[1].metric("TPSA", p.get("TPSA", "—"))
            metric_cols[0].metric("H-bond donors", p.get("HBondDonorCount", "—"))
            metric_cols[1].metric("H-bond acceptors", p.get("HBondAcceptorCount", "—"))
        st.subheader("Interactive 3D structure")
        style = st.selectbox("Display style", ["Ball and stick", "Stick", "Space filling", "Wireframe"])
        surface = st.checkbox("Show van der Waals surface")
        if st.button("Load 3D model", key="load_3d"):
            with st.spinner("Loading the PubChem 3D conformer..."):
                display_3d(record["cid"], style, surface)
        with st.expander("Show complete descriptor table"):
            st.dataframe(pd.DataFrame(property_rows(record), columns=["Property", "Value"]), use_container_width=True, hide_index=True)

# Property report
elif page == "📊 Property Report":
    st.title("📊 Molecular Property Report")
    record = st.session_state.get("active_record")
    if not record:
        st.info("Analyze a molecule in the Molecular Explorer first.")
    else:
        st.subheader(f"Report for PubChem CID {record['cid']}")
        df = pd.DataFrame(property_rows(record), columns=["Property", "Value"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "molecular_properties.csv", "text/csv")
        st.download_button("📄 Download PDF report", make_pdf(record, selected_student, STUDENTS[selected_student]["reg"]), "molecular_analysis_report.pdf", "application/pdf")
        numeric = {}
        for key in ["MolecularWeight", "ExactMass", "XLogP", "TPSA", "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount", "HeavyAtomCount"]:
            value = record["properties"].get(key)
            if isinstance(value, (int, float)):
                numeric[key] = float(value)
        if numeric:
            st.subheader("Descriptor overview")
            fig, ax = plt.subplots(figsize=(9, 3.8))
            ax.bar(list(numeric.keys()), list(numeric.values()))
            ax.set_ylabel("Reported value")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# Comparison
elif page == "📈 Molecule Comparison":
    st.title("📈 Molecule Comparison")
    st.write("Enter one molecule per line using the format `Name,SMILES`. Up to six compounds are supported per comparison.")
    default_lines = "\n".join(f"{name},{smiles}" for name, smiles in list(EXAMPLES.items())[:5])
    text_input = st.text_area("Molecules", value=default_lines, height=180)
    if st.button("Compare molecules", type="primary"):
        rows, errors = [], []
        entries = [line.strip() for line in text_input.splitlines() if line.strip()]
        if len(entries) > 6:
            st.warning("Only the first six entries will be processed.")
            entries = entries[:6]
        for line in entries:
            if "," not in line:
                errors.append(f"Missing comma: {line}")
                continue
            name, smiles_value = line.split(",", 1)
            try:
                rec = analyze_smiles(smiles_value.strip())
                p = rec["properties"]
                rows.append({
                    "Molecule": name.strip(), "CID": rec["cid"],
                    "Formula": p.get("MolecularFormula"),
                    "Molecular weight": p.get("MolecularWeight"),
                    "XLogP": p.get("XLogP"), "TPSA": p.get("TPSA"),
                    "HBD": p.get("HBondDonorCount"), "HBA": p.get("HBondAcceptorCount"),
                    "Rotatable bonds": p.get("RotatableBondCount"),
                })
            except (ValueError, requests.RequestException) as exc:
                errors.append(f"{name.strip()}: {exc}")
        if rows:
            comparison = pd.DataFrame(rows)
            st.session_state["comparison_df"] = comparison
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download comparison CSV", comparison.to_csv(index=False).encode("utf-8"), "molecule_comparison.csv", "text/csv")
            chart_field = st.selectbox("Plot descriptor", ["Molecular weight", "XLogP", "TPSA", "HBD", "HBA", "Rotatable bonds"])
            chart_data = comparison.dropna(subset=[chart_field])
            if not chart_data.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(chart_data["Molecule"], pd.to_numeric(chart_data[chart_field], errors="coerce"))
                ax.set_ylabel(chart_field)
                ax.tick_params(axis="x", rotation=25)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        for error in errors:
            st.warning(error)
        if not rows and not errors:
            st.info("Enter at least one molecule to compare.")

# Assessment
elif page == "📝 Assessment":
    st.title("📝 Cheminformatics Assessment")
    questions = [
        ("What does SMILES represent?", ["A text representation of molecular structure", "A laboratory instrument", "A spectroscopy method"], 0),
        ("Which SMILES is ethanol?", ["c1ccccc1", "CCO", "CC(=O)O"], 1),
        ("Which SMILES is benzene?", ["c1ccccc1", "CCO", "O=C=O"], 0),
        ("What does XLogP estimate?", ["Molecular colour", "Lipophilicity", "Boiling point only"], 1),
        ("TPSA stands for:", ["Topological Polar Surface Area", "Total Protein Structure Analysis", "Thermal Pressure Scale Average"], 0),
        ("HBD means:", ["Heavy Bond Descriptor", "Hydrogen Bond Donor", "Hydrocarbon Bond Density"], 1),
        ("HBA means:", ["Hydrogen Bond Acceptor", "Heavy Base Atom", "Hydrogen Bond Angle"], 0),
        ("A double bond contains how many pi bonds?", ["0", "1", "2"], 1),
        ("A triple bond contains how many pi bonds?", ["1", "2", "3"], 1),
        ("What does a 2D molecular drawing show?", ["Atom connectivity", "Only molecular weight", "Reaction temperature"], 0),
    ]
    with st.form("assessment_form"):
        answers = []
        for index, (question, options, correct) in enumerate(questions):
            answers.append(st.radio(f"{index + 1}. {question}", options, index=None, key=f"quiz_{index}"))
        submitted = st.form_submit_button("Submit assessment")
    if submitted:
        if any(answer is None for answer in answers):
            st.warning("Please answer every question before submitting.")
        else:
            score = sum(answer == questions[i][1][questions[i][2]] for i, answer in enumerate(answers))
            st.session_state["quiz_result"] = (score, len(questions))
            st.success(f"Your score: {score}/{len(questions)} ({score / len(questions) * 100:.0f}%)")
            if score >= 8:
                st.balloons()
                st.success("Excellent work! You have a strong understanding of the basics.")
            elif score >= 5:
                st.info("Good effort. Review the theory and try again.")
            else:
                st.info("Keep learning. Revisit the theory section and practice with examples.")

# Learning resources
elif page == "📚 Learning Resources":
    st.title("📚 Learning Resources")
    st.markdown("### Quick SMILES examples")
    st.dataframe(pd.DataFrame([{"Compound": k, "SMILES": v} for k, v in EXAMPLES.items()]), use_container_width=True, hide_index=True)
    st.markdown("### Data and documentation")
    st.markdown("- [PubChem database](https://pubchem.ncbi.nlm.nih.gov/)\n- [PubChem PUG REST documentation](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)\n- [Streamlit documentation](https://docs.streamlit.io/)")
    st.info("This version does not require RDKit. It uses PubChem's online records, so an internet connection is required for compound lookups, images, and 3D structures.")

st.divider()
st.caption("SRM Cheminformatics Virtual Laboratory · Student project by Harini M and Harini Sri · Powered by Streamlit and PubChem PUG REST")
