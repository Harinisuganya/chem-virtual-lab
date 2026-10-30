import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# RDKit
from rdkit import Chem
from rdkit.Chem import (
    Draw,
    AllChem,
    Descriptors,
    Crippen,
    Lipinski,
    rdMolDescriptors
)

# 3D Visualization
import py3Dmol
import streamlit.components.v1 as components

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    Preformatted
)
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
import tempfile
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cheminformatics Virtual Laboratory",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}

.section-title {
    font-size: 25px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

# ============================================================
# SHARED MOLECULE INPUT STATE
# ============================================================

EXAMPLE_MOLECULES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic Acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
}

if "active_smiles" not in st.session_state:
    st.session_state.active_smiles = "CCO"

if "selected_example" not in st.session_state:
    st.session_state.selected_example = "Ethanol"

def load_shared_example():
    """Load the selected example into the shared SMILES field."""
    st.session_state.active_smiles = EXAMPLE_MOLECULES[
        st.session_state.selected_example
    ]

def get_molecule(smiles):
    """Convert SMILES into an RDKit molecule."""

    if not smiles or not smiles.strip():
        return None

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol

    except Exception:
        return None


# ============================================================
# CALCULATE SIGMA AND PI BONDS
# ============================================================

def calculate_bond_information(mol):
    """
    Calculate:
    - Heavy-atom bonds
    - Sigma (σ) bonds, including bonds to implicit hydrogens
    - Pi (π) bonds
    - Total bond order count (σ + π)
    """

    # Heavy-atom graph bonds (RDKit default molecule representation)
    heavy_atom_bonds = mol.GetNumBonds()

    # Count π bonds using a Kekulé representation for aromatic systems
    kekule_mol = Chem.Mol(mol)

    try:
        Chem.Kekulize(kekule_mol, clearAromaticFlags=True)
    except Exception:
        # If Kekulization is not possible, continue with the available form
        pass

    pi_bonds = 0

    for bond in kekule_mol.GetBonds():
        bond_type = bond.GetBondType()

        if bond_type == Chem.BondType.DOUBLE:
            pi_bonds += 1
        elif bond_type == Chem.BondType.TRIPLE:
            pi_bonds += 2
        elif bond_type == Chem.BondType.QUADRUPLE:
            pi_bonds += 3
        elif bond.GetIsAromatic():
            # Fallback for aromatic bonds if Kekulization did not clear them
            pi_bonds += 0.5

    # Add implicit hydrogens. Every atom-to-atom connection contains one σ bond.
    mol_with_h = Chem.AddHs(Chem.Mol(mol))
    sigma_bonds = mol_with_h.GetNumBonds()

    total_bonds = sigma_bonds + pi_bonds

    return {
        "Heavy Atom Bonds": heavy_atom_bonds,
        "Sigma Bonds": sigma_bonds,
        "Pi Bonds": pi_bonds,
        "Total Bonds": total_bonds,
        "Total Atoms Including H": mol_with_h.GetNumAtoms(),
    }


# ============================================================
# CALCULATE COMPLETE MOLECULAR PROPERTIES
# ============================================================

def calculate_properties(mol):

    # Calculate bond information
    bond_info = calculate_bond_information(mol)

    molecular_formula = rdMolDescriptors.CalcMolFormula(mol)

    properties = {

        "Molecular Formula": molecular_formula,

        "Molecular Weight": round(
            Descriptors.MolWt(mol),
            3
        ),

        "Exact Molecular Weight": round(
            Descriptors.ExactMolWt(mol),
            4
        ),

        "LogP": round(
            Crippen.MolLogP(mol),
            3
        ),

        "TPSA": round(
            rdMolDescriptors.CalcTPSA(mol),
            3
        ),

        "Hydrogen Bond Donors": Lipinski.NumHDonors(mol),

        "Hydrogen Bond Acceptors": Lipinski.NumHAcceptors(mol),

        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),

        "Ring Count": Lipinski.RingCount(mol),

        "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),

        "Aliphatic Rings": rdMolDescriptors.CalcNumAliphaticRings(mol),

        "Number of Atoms": mol.GetNumAtoms(),

        "Heavy Atoms": mol.GetNumHeavyAtoms(),

        # Bond information
        "Heavy Atom Bonds": bond_info["Heavy Atom Bonds"],

        "Sigma (σ) Bonds": bond_info["Sigma Bonds"],

        "Pi (π) Bonds": bond_info["Pi Bonds"],

        "Total Bonds (σ + π)": bond_info["Total Bonds"],

        "Fraction Csp3": round(
            rdMolDescriptors.CalcFractionCSP3(mol),
            3
        ),

        "Molar Refractivity": round(
            Crippen.MolMR(mol),
            3
        )
    }

    return properties

# ============================================================
# GENERATE 2D IMAGE
# ============================================================

def generate_2d_image(mol):

    image = Draw.MolToImage(
        mol,
        size=(700, 500)
    )

    return image


# ============================================================
# GENERATE 3D MOLECULE
# ============================================================

def generate_3d_molecule(mol):

    try:

        mol3d = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()

        params.randomSeed = 42

        status = AllChem.EmbedMolecule(
            mol3d,
            params
        )

        if status != 0:
            return None

        try:

            AllChem.MMFFOptimizeMolecule(
                mol3d
            )

        except Exception:

            try:

                AllChem.UFFOptimizeMolecule(
                    mol3d
                )

            except Exception:
                pass

        return mol3d

    except Exception:

        return None


# ============================================================
# DISPLAY 3D STRUCTURE
# ============================================================

def display_3d_structure(mol3d, style, show_surface=False):

    mol_block = Chem.MolToMolBlock(mol3d)

    view = py3Dmol.view(
        width=900,
        height=600
    )

    view.addModel(
        mol_block,
        "mol"
    )

    # --------------------------------------------------------
    # BALL AND STICK
    # --------------------------------------------------------

    if style == "Ball and Stick":

        view.setStyle({

            "stick": {
                "radius": 0.15
            },

            "sphere": {
                "scale": 0.30
            }

        })

    # --------------------------------------------------------
    # STICK
    # --------------------------------------------------------

    elif style == "Stick":

        view.setStyle({

            "stick": {
                "radius": 0.20
            }

        })

    # --------------------------------------------------------
    # SPACE FILLING
    # --------------------------------------------------------

    elif style == "Space Filling":

        view.setStyle({

            "sphere": {
                "scale": 1.0
            }

        })

    # --------------------------------------------------------
    # WIREFRAME
    # --------------------------------------------------------

    elif style == "Wireframe":

        view.setStyle({

            "line": {
                "linewidth": 2
            }

        })

    # --------------------------------------------------------
    # CARTOON STYLE
    # --------------------------------------------------------

    elif style == "Cartoon":

        # Cartoon is mainly useful for biomolecules,
        # but this provides a molecular representation.

        view.setStyle({

            "stick": {},

            "sphere": {
                "scale": 0.20
            }

        })

    # --------------------------------------------------------
    # OPTIONAL SURFACE
    # --------------------------------------------------------

    if show_surface:

        view.addSurface(
            py3Dmol.VDW,
            {
                "opacity": 0.7
            }
        )

    view.setBackgroundColor("white")

    view.zoomTo()

    html = view._make_html()

    components.html(
        html,
        height=620
    )


# ============================================================
# CREATE MOLECULAR PDF REPORT
# ============================================================

def generate_molecule_pdf(
    input_smiles,
    mol,
    properties
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=45,
        rightMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=20,
        alignment=TA_CENTER,
        textColor=HexColor("#17365D")
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=HexColor("#17365D")
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY
    )

    story = []

    story.append(
        Paragraph(
            "MOLECULAR PROPERTIES REPORT",
            title_style
        )
    )

    story.append(
        Spacer(1, 0.3 * inch)
    )

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Molecular Information",
            heading_style
        )
    )

    canonical_smiles = Chem.MolToSmiles(mol)

    basic_data = [

        ["Property", "Value"],

        ["Input SMILES", input_smiles],

        ["Canonical SMILES", canonical_smiles],

        [
            "Molecular Formula",
            properties["Molecular Formula"]
        ]
    ]

    basic_table = Table(
        basic_data,
        colWidths=[2 * inch, 4 * inch]
    )

    basic_table.setStyle(
        TableStyle([

            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#17365D")),

            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

            ("VALIGN", (0, 0), (-1, -1), "TOP"),

            ("PADDING", (0, 0), (-1, -1), 7)

        ])
    )

    story.append(basic_table)

    story.append(
        Spacer(1, 0.3 * inch)
    )

    # --------------------------------------------------------
    # PROPERTY TABLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Molecular Properties",
            heading_style
        )
    )

    property_data = [
        ["Property", "Value"]
    ]

    for key, value in properties.items():

        property_data.append(
            [key, str(value)]
        )

    property_table = Table(
        property_data,
        colWidths=[3 * inch, 3 * inch],
        repeatRows=1
    )

    property_table.setStyle(
        TableStyle([

            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),

            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

            ("VALIGN", (0, 0), (-1, -1), "TOP"),

            ("PADDING", (0, 0), (-1, -1), 6)

        ])
    )

    story.append(property_table)

    story.append(
        Spacer(1, 0.3 * inch)
    )

    story.append(
        Paragraph(
            "3. Interpretation",
            heading_style
        )
    )

    interpretation = """
    This report was generated automatically using the
    Cheminformatics Virtual Laboratory. The molecular properties
    were calculated from the chemical structure represented by
    the submitted SMILES notation using the RDKit cheminformatics
    toolkit.
    """

    story.append(
        Paragraph(
            interpretation,
            body_style
        )
    )

    # --------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------

    doc.build(story)

    pdf_data = buffer.getvalue()

    buffer.close()

    return pdf_data


# ============================================================
# REPORT IMAGE HELPERS
# ============================================================

def save_2d_structure_png(mol):
    """Create a temporary PNG for inclusion in PDF reports."""
    img = Draw.MolToImage(mol, size=(700, 500))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    img.save(tmp.name)
    return tmp.name


def save_3d_structure_png(mol):
    """
    Create a high-clarity ball-and-stick image for the PDF report.

    The RDKit 3D coordinates are projected onto the best viewing plane so
    elongated molecules fill the page instead of appearing as a tiny object.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    mol3d = generate_3d_molecule(mol)
    if mol3d is None:
        return None

    conf = mol3d.GetConformer()
    n_atoms = mol3d.GetNumAtoms()

    # Collect 3D coordinates.
    coords = np.array([
        [
            conf.GetAtomPosition(i).x,
            conf.GetAtomPosition(i).y,
            conf.GetAtomPosition(i).z,
        ]
        for i in range(n_atoms)
    ], dtype=float)

    # Principal-component projection chooses the clearest overall view.
    centered = coords - coords.mean(axis=0)
    if n_atoms >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:2].T
    else:
        projected = centered[:, :2]

    # CPK-style colours and readable atom sizes.
    cpk = {
        "H": "#FFFFFF", "C": "#3F3F3F", "N": "#2F5BFF",
        "O": "#E53935", "F": "#65C466", "P": "#F39C12",
        "S": "#F4D03F", "Cl": "#27AE60", "Br": "#8E3B2F",
        "I": "#8E44AD",
    }
    sizes = {
        "H": 70, "C": 240, "N": 270, "O": 285, "F": 260,
        "P": 300, "S": 310, "Cl": 310, "Br": 330, "I": 350,
    }

    fig, ax = plt.subplots(figsize=(8.8, 6.4), facecolor="white")
    ax.set_facecolor("white")

    # Draw bonds first. Multiple bonds are shown with parallel lines.
    for bond in mol3d.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        x1, y1 = projected[i]
        x2, y2 = projected[j]

        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length == 0:
            continue

        # Perpendicular direction for double/triple bond separation.
        px, py = -dy / length, dx / length
        btype = bond.GetBondType()
        if btype == Chem.BondType.DOUBLE:
            offsets = (-0.035, 0.035)
        elif btype == Chem.BondType.TRIPLE:
            offsets = (-0.055, 0.0, 0.055)
        else:
            offsets = (0.0,)

        for off in offsets:
            ax.plot(
                [x1 + px * off, x2 + px * off],
                [y1 + py * off, y2 + py * off],
                color="#707070", linewidth=3.0,
                solid_capstyle="round", zorder=1
            )

    # Draw atoms over bonds.
    for i, atom in enumerate(mol3d.GetAtoms()):
        symbol = atom.GetSymbol()
        x, y = projected[i]
        ax.scatter(
            x, y,
            s=sizes.get(symbol, 250),
            c=cpk.get(symbol, "#A0A0A0"),
            edgecolors="#222222",
            linewidths=1.0,
            zorder=3
        )

        # Label important/non-carbon hetero atoms only; this keeps the model clean.
        if symbol not in ("H", "C"):
            ax.text(
                x, y, symbol,
                ha="center", va="center",
                fontsize=8, fontweight="bold",
                color="black", zorder=4
            )

    # Tight, equal-aspect limits make the molecule large and clear.
    xmin, ymin = projected.min(axis=0)
    xmax, ymax = projected.max(axis=0)
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    pad = 0.22 * max(width, height)

    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title("3D Molecular Structure (Ball-and-Stick)", fontsize=18, fontweight="bold", pad=18)

    fig.tight_layout(pad=1.2)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(
        tmp.name,
        dpi=320,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.12
    )
    plt.close(fig)
    return tmp.name


# ============================================================
# PROPERTY INTERPRETATION HELPERS
# ============================================================

def get_property_interpretation(property_name, value, properties):
    """Return a student-friendly interpretation for each reported property."""

    if property_name == "Molecular Formula":
        return (
            "Shows the elemental composition of the molecule and is useful "
            "for understanding its basic chemical composition."
        )

    if property_name == "Molecular Weight":
        return (
            "Represents the average molecular mass. Molecular size and mass "
            "can influence diffusion, formulation and many physicochemical behaviours."
        )

    if property_name == "Exact Molecular Weight":
        return (
            "The monoisotopic mass calculated from exact isotopic masses; "
            "this is especially useful when comparing mass-spectrometry data."
        )

    if property_name == "LogP":
        if value < 0:
            return "A negative value indicates a preference toward the aqueous phase and relatively high hydrophilicity."
        elif value < 1:
            return "Indicates low to moderate lipophilicity, suggesting a reasonable preference for polar environments."
        elif value < 3:
            return "Indicates moderate lipophilicity, often reflecting a balance between hydrophilic and hydrophobic character."
        else:
            return "Indicates relatively high lipophilicity and a stronger preference for non-polar environments."

    if property_name == "TPSA":
        if value < 60:
            return "A relatively low polar surface area, indicating limited overall molecular polarity."
        elif value < 140:
            return "A moderate polar surface area, reflecting the contribution of heteroatoms to molecular polarity."
        else:
            return "A high polar surface area, indicating substantial polarity and strong hydrogen-bonding potential."

    if property_name == "Hydrogen Bond Donors":
        return (
            f"The molecule has {value} donor site(s) capable of donating hydrogen in hydrogen-bond interactions."
        )

    if property_name == "Hydrogen Bond Acceptors":
        return (
            f"The molecule has {value} acceptor site(s) capable of accepting hydrogen in hydrogen-bond interactions."
        )

    if property_name == "Rotatable Bonds":
        if value == 0:
            return "No freely rotatable bonds are detected, suggesting a relatively rigid molecular framework."
        return f"{value} rotatable bond(s) indicate conformational flexibility in the molecular structure."

    if property_name == "Ring Count":
        return f"The molecule contains {value} ring(s), which contribute to its overall shape and structural rigidity."

    if property_name == "Aromatic Rings":
        return f"{value} aromatic ring(s) contribute to aromaticity, planarity and delocalized π-electron character."

    if property_name == "Aliphatic Rings":
        return f"{value} non-aromatic ring(s) contribute to cyclic structure without aromatic π-electron delocalization."

    if property_name == "Number of Atoms":
        return "Counts atoms explicitly present in the RDKit molecular representation; implicit hydrogens are not included here."

    if property_name == "Heavy Atoms":
        return "Counts all non-hydrogen atoms and provides a useful measure of molecular structural size."

    if property_name == "Heavy Atom Bonds":
        return "Counts connections between non-hydrogen atoms in the molecular graph."

    if property_name == "Sigma (σ) Bonds":
        return "Every bonded atom pair contains one σ bond; this value includes bonds involving explicit or added hydrogens."

    if property_name == "Pi (π) Bonds":
        return "π bonds arise from multiple bonds and aromatic systems and are associated with electron delocalization."

    if property_name == "Total Bonds (σ + π)":
        return "Represents the total bond components obtained by adding the calculated σ and π bond counts."

    if property_name == "Fraction Csp3":
        if value < 0.25:
            return "A low sp³ fraction suggests a comparatively planar, unsaturated or aromatic structural character."
        elif value < 0.60:
            return "An intermediate sp³ fraction indicates a balance between saturated and unsaturated structural features."
        return "A high sp³ fraction indicates a more saturated, three-dimensional molecular framework."

    if property_name == "Molar Refractivity":
        return "Relates to molecular volume and electronic polarizability and reflects how the electron cloud responds to an electric field."

    return "This descriptor provides additional quantitative information about the molecular structure and physicochemical behaviour."


def build_key_interpretation_points(properties):
    """Create concise point-wise conclusions for the final report."""
    points = []

    points.append(
        f"<b>• Molecular identity:</b> The submitted structure has molecular formula "
        f"<b>{properties['Molecular Formula']}</b> and molecular weight "
        f"<b>{properties['Molecular Weight']}</b>."
    )

    logp = properties["LogP"]
    if logp < 0:
        lipophilic_text = "more hydrophilic than lipophilic"
    elif logp < 1:
        lipophilic_text = "low in lipophilicity"
    elif logp < 3:
        lipophilic_text = "moderately lipophilic"
    else:
        lipophilic_text = "strongly lipophilic"

    points.append(
        f"<b>• Lipophilicity:</b> LogP = <b>{logp}</b>, indicating that the molecule is "
        f"<b>{lipophilic_text}</b>. This descriptor describes the balance between "
        f"preference for non-polar and polar environments."
    )

    points.append(
        f"<b>• Molecular polarity:</b> TPSA = <b>{properties['TPSA']}</b> Å². "
        f"TPSA reflects the contribution of polar atoms and functional groups to the "
        f"overall surface polarity of the molecule."
    )

    points.append(
        f"<b>• Hydrogen bonding:</b> The structure contains "
        f"<b>{properties['Hydrogen Bond Donors']}</b> hydrogen-bond donor(s) and "
        f"<b>{properties['Hydrogen Bond Acceptors']}</b> acceptor(s), which helps "
        f"describe possible intermolecular interactions."
    )

    points.append(
        f"<b>• Flexibility:</b> {properties['Rotatable Bonds']} rotatable bond(s) "
        f"indicate the degree of conformational freedom available to the molecule."
    )

    points.append(
        f"<b>• Cyclic and aromatic character:</b> The molecule contains "
        f"{properties['Ring Count']} ring(s), including "
        f"{properties['Aromatic Rings']} aromatic ring(s). Aromatic rings are "
        f"associated with delocalized π-electron systems."
    )

    points.append(
        f"<b>• Bonding pattern:</b> The calculated structure contains "
        f"<b>{properties['Sigma (σ) Bonds']}</b> σ bonds and "
        f"<b>{properties['Pi (π) Bonds']}</b> π bonds. This helps distinguish the "
        f"framework-forming σ bonds from multiple-bond and delocalized π bonding."
    )

    points.append(
        f"<b>• Three-dimensional character:</b> Fraction Csp³ = "
        f"<b>{properties['Fraction Csp3']}</b>. This descriptor provides a simple "
        f"indication of the balance between saturated 3D carbon environments and "
        f"planar/unsaturated environments."
    )

    points.append(
        "<b>• Overall conclusion:</b> The calculated descriptors should be interpreted "
        "together with the 2D connectivity and generated 3D geometry to understand "
        "the molecule's composition, bonding, polarity, flexibility and structural character."
    )

    return points


# ============================================================
# GENERATE GENERAL FINAL PROJECT REPORT
# ============================================================

def generate_final_report(student_name, registration_number, input_smiles, mol, properties):
    """Generate a detailed personalized final report for the student's molecule."""

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=38,
        rightMargin=38,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ProjectTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=25,
        alignment=TA_CENTER,
        textColor=HexColor("#17365D")
    )

    heading_style = ParagraphStyle(
        "ProjectHeading",
        parent=styles["Heading1"],
        fontSize=14,
        leading=18,
        textColor=HexColor("#17365D"),
        spaceBefore=8,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "ProjectBody",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
        alignment=TA_JUSTIFY
    )

    small_style = ParagraphStyle(
        "SmallTableText",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=9.5
    )

    header_style = ParagraphStyle(
        "HeaderTableText",
        parent=styles["BodyText"],
        fontSize=8,
        leading=9,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    story = []

    story += [
        Spacer(1, 0.25 * inch),
        Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", title_style),
        Spacer(1, 0.12 * inch),
        Paragraph("FINAL STUDENT MOLECULAR ANALYSIS REPORT", heading_style),
        Spacer(1, 0.18 * inch)
    ]

    student_data = [
        ["Student Name", student_name or "Not provided"],
        ["Registration Number", registration_number or "Not provided"],
        ["Input SMILES", input_smiles],
        ["Canonical SMILES", Chem.MolToSmiles(mol)],
        ["Molecular Formula", properties["Molecular Formula"]]
    ]

    student_table = Table(student_data, colWidths=[2.0 * inch, 4.7 * inch])
    student_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), HexColor("#EAF0F8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))

    story += [student_table, Spacer(1, 0.22 * inch)]

    # --------------------------------------------------------
    # 1. STRUCTURES
    # --------------------------------------------------------
    story.append(Paragraph("1. Molecular Structures", heading_style))

    img2d = save_2d_structure_png(mol)
    img3d = save_3d_structure_png(mol)

    if img2d and img3d:
        structure_table = Table(
            [[
                Image(img2d, width=3.15 * inch, height=2.25 * inch),
                Image(img3d, width=3.15 * inch, height=2.25 * inch)
            ]],
            colWidths=[3.3 * inch, 3.3 * inch]
        )

        structure_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER")
        ]))

        story += [
            structure_table,
            Spacer(1, 0.08 * inch),
            Paragraph(
                "<b>Figure 1.</b> Left: 2D molecular connectivity. "
                "Right: generated 3D ball-and-stick geometry showing the spatial arrangement of atoms.",
                body_style
            )
        ]

    elif img2d:
        story.append(Image(img2d, width=5.8 * inch, height=4.1 * inch))

    story += [Spacer(1, 0.18 * inch)]

    # --------------------------------------------------------
    # 2. COMPLETE PROPERTIES WITH INTERPRETATION COLUMN
    # --------------------------------------------------------
    story.append(Paragraph("2. Complete Molecular Properties and Interpretation", heading_style))

    prop_data = [[
        Paragraph("<b>Property</b>", header_style),
        Paragraph("<b>Value</b>", header_style),
        Paragraph("<b>Interpretation</b>", header_style)
    ]]

    for key, value in properties.items():
        interpretation = get_property_interpretation(key, value, properties)
        prop_data.append([
            Paragraph(str(key), small_style),
            Paragraph(str(value), small_style),
            Paragraph(interpretation, small_style)
        ])

    property_table = Table(
        prop_data,
        colWidths=[1.75 * inch, 1.0 * inch, 3.85 * inch],
        repeatRows=1
    )

    property_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4)
    ]))

    story += [property_table, Spacer(1, 0.18 * inch)]

    # --------------------------------------------------------
    # 3. POINT-WISE INTERPRETATION
    # --------------------------------------------------------
    story.append(Paragraph("3. Point-wise Molecular Interpretation", heading_style))

    intro = (
        "The following points summarize the most important structural and "
        "physicochemical features calculated from the student's submitted SMILES."
    )
    story.append(Paragraph(intro, body_style))
    story.append(Spacer(1, 0.08 * inch))

    for point in build_key_interpretation_points(properties):
        story.append(Paragraph(point, body_style))
        story.append(Spacer(1, 0.06 * inch))

    # --------------------------------------------------------
    # 4. LEARNING NOTE
    # --------------------------------------------------------
    story += [
        Spacer(1, 0.12 * inch),
        Paragraph("4. Learning Note", heading_style),
        Paragraph(
            "The numerical descriptors in this report are computationally calculated "
            "from the submitted molecular structure. They are useful for molecular "
            "comparison and cheminformatics learning, but experimental measurements "
            "may differ depending on conditions and methodology.",
            body_style
        )
    ]


    # --------------------------------------------------------
    # 5. PYTHON LIBRARIES AND THEIR IMPORTANCE
    # --------------------------------------------------------
    story += [
        Spacer(1, 0.12 * inch),
        Paragraph("5. Python Libraries Used and Their Importance", heading_style),
        Paragraph(
            "The virtual laboratory uses the following libraries to process "
            "molecular structures, calculate descriptors, display results and "
            "create this PDF report.",
            body_style
        ),
        Spacer(1, 0.08 * inch)
    ]

    library_rows = [
        [
            Paragraph("<b>Library</b>", header_style),
            Paragraph("<b>Importance in this project</b>", header_style)
        ],
        [
            Paragraph("Streamlit", small_style),
            Paragraph("Builds the interactive web application, input fields, navigation, "
                      "visualization controls and download buttons.", small_style)
        ],
        [
            Paragraph("RDKit (Chem, Descriptors, Crippen, Lipinski, rdMolDescriptors, Draw, AllChem)",
                      small_style),
            Paragraph("Reads SMILES, validates molecular structures, generates 2D drawings "
                      "and 3D coordinates, and calculates molecular formula, molecular weight, "
                      "LogP, TPSA, hydrogen-bond counts, ring counts and other descriptors.",
                      small_style)
        ],
        [
            Paragraph("Pandas", small_style),
            Paragraph("Organizes molecular properties into tables and supports CSV export.",
                      small_style)
        ],
        [
            Paragraph("Matplotlib", small_style),
            Paragraph("Creates a clear 2D projection of the generated 3D molecular geometry "
                      "for inclusion in the PDF report.", small_style)
        ],
        [
            Paragraph("NumPy", small_style),
            Paragraph("Processes 3D atomic coordinates and uses numerical linear algebra "
                      "to select a clear projection of the molecule.", small_style)
        ],
        [
            Paragraph("py3Dmol", small_style),
            Paragraph("Displays interactive 3D molecular models in the web application.",
                      small_style)
        ],
        [
            Paragraph("ReportLab", small_style),
            Paragraph("Creates the downloadable PDF, including student details, structures, "
                      "property tables, interpretations, code examples and library notes.",
                      small_style)
        ],
        [
            Paragraph("Python standard libraries: io, os, tempfile", small_style),
            Paragraph("Manage in-memory PDF data and temporary image files, and clean up "
                      "temporary files after report generation.", small_style)
        ]
    ]

    library_table = Table(
        library_rows,
        colWidths=[2.05 * inch, 4.65 * inch],
        repeatRows=1
    )
    library_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5)
    ]))
    story += [library_table, Spacer(1, 0.18 * inch)]

    # --------------------------------------------------------
    # 6. ILLUSTRATIVE PYTHON CODE
    # --------------------------------------------------------
    story.append(Paragraph("6. Python Code: SMILES to Molecular Properties", heading_style))
    story.append(Paragraph(
        "The following educational example shows the core workflow used by this "
        "laboratory. It is a compact example of the main calculation steps, not the "
        "complete Streamlit application source code.",
        body_style
    ))
    story.append(Spacer(1, 0.08 * inch))

    code_style = ParagraphStyle(
        "EducationalCode",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7,
        leading=8.5,
        leftIndent=5,
        rightIndent=5,
        spaceBefore=4,
        spaceAfter=4
    )

    educational_code = '''from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, Draw
from rdkit.Chem import AllChem

# 1. Convert a SMILES string into a molecular object
smiles = "CCO"  # Example: ethanol
mol = Chem.MolFromSmiles(smiles)

if mol is None:
    raise ValueError("Invalid SMILES notation")

# 2. Calculate selected molecular descriptors
formula = rdMolDescriptors.CalcMolFormula(mol)
molecular_weight = Descriptors.MolWt(mol)
logp = Crippen.MolLogP(mol)
tpsa = rdMolDescriptors.CalcTPSA(mol)
h_bond_donors = Lipinski.NumHDonors(mol)
h_bond_acceptors = Lipinski.NumHAcceptors(mol)

print("Formula:", formula)
print("Molecular weight:", round(molecular_weight, 3), "g/mol")
print("LogP:", round(logp, 3))
print("TPSA:", round(tpsa, 3), "A^2")
print("H-bond donors:", h_bond_donors)
print("H-bond acceptors:", h_bond_acceptors)

# 3. Save a 2D molecular structure image
Draw.MolToFile(mol, "molecule_2d.png", size=(700, 500))

# 4. Generate a 3D conformer (coordinates)
mol_3d = Chem.AddHs(mol)
status = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
if status == 0:
    AllChem.MMFFOptimizeMolecule(mol_3d)
    print("3D coordinates generated successfully")'''

    story.append(Preformatted(educational_code, code_style, maxLineLength=92))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("How to read the code", heading_style))
    code_notes = [
        "<b>SMILES input:</b> A text representation of the molecular connectivity.",
        "<b>Validation:</b> <font name='Courier'>Chem.MolFromSmiles()</font> attempts to parse the input; "
        "an invalid structure returns <font name='Courier'>None</font>.",
        "<b>Descriptors:</b> RDKit functions calculate numerical properties from the parsed molecular graph.",
        "<b>2D drawing:</b> <font name='Courier'>Draw.MolToFile()</font> writes a molecular diagram to an image file.",
        "<b>3D geometry:</b> Explicit hydrogens are added, a conformer is generated, and MMFF optimization "
        "is attempted to refine its geometry."
    ]
    for note in code_notes:
        story.append(Paragraph("• " + note, body_style))
        story.append(Spacer(1, 0.04 * inch))

    # --------------------------------------------------------
    # 7. REPRODUCIBILITY NOTE
    # --------------------------------------------------------
    story += [
        Spacer(1, 0.10 * inch),
        Paragraph("7. Reproducibility and Limitations", heading_style),
        Paragraph(
            "This code is an educational illustration of the principal workflow. "
            "The web application contains additional functions for student inputs, "
            "property interpretation, 3D display and PDF formatting. Calculated "
            "descriptors are computational estimates and should be interpreted with "
            "their definitions and, where appropriate, compared with experimental data.",
            body_style
        )
    ]


    doc.build(story)

    pdf_data = buffer.getvalue()
    buffer.close()

    for path in [img2d, img3d]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    return pdf_data



# ============================================================
# GENERATE ONE PDF FOR MULTIPLE MOLECULES
# ============================================================

def generate_multi_final_report(student_name, registration_number, molecule_records):
    """Create one PDF containing results for every valid SMILES supplied."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=38, rightMargin=38,
        topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MultiReportTitle", parent=styles["Title"], fontSize=19,
        leading=23, alignment=TA_CENTER, textColor=HexColor("#17365D")
    )
    heading_style = ParagraphStyle(
        "MultiReportHeading", parent=styles["Heading1"], fontSize=13,
        leading=16, textColor=HexColor("#17365D"), spaceBefore=7, spaceAfter=7
    )
    body_style = ParagraphStyle(
        "MultiReportBody", parent=styles["BodyText"], fontSize=9,
        leading=12, alignment=TA_JUSTIFY
    )
    small_style = ParagraphStyle(
        "MultiReportSmall", parent=styles["BodyText"], fontSize=7.2, leading=8.8
    )
    header_style = ParagraphStyle(
        "MultiReportHeader", parent=styles["BodyText"], fontSize=7.5,
        leading=9, textColor=colors.white, alignment=TA_CENTER
    )
    story = [
        Spacer(1, 0.15 * inch),
        Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY", title_style),
        Spacer(1, 0.08 * inch),
        Paragraph("MULTI-MOLECULE STUDENT ANALYSIS REPORT", heading_style),
        Paragraph(
            f"Student: {student_name or 'Not provided'} &nbsp;&nbsp;&nbsp; "
            f"Registration Number: {registration_number or 'Not provided'}",
            body_style
        ),
        Paragraph(f"Number of molecules analyzed: {len(molecule_records)}", body_style),
        Spacer(1, 0.15 * inch)
    ]

    temp_images = []
    for index, record in enumerate(molecule_records, start=1):
        input_smiles, mol, properties = record
        if index > 1:
            story.append(PageBreak())

        story.append(Paragraph(f"Molecule {index}: Molecular Information", heading_style))
        basic_data = [
            ["Input SMILES", input_smiles],
            ["Canonical SMILES", Chem.MolToSmiles(mol)],
            ["Molecular Formula", properties["Molecular Formula"]],
            ["Molecular Weight", str(properties["Molecular Weight"]) + " g/mol"]
        ]
        basic_table = Table(basic_data, colWidths=[1.8 * inch, 4.9 * inch])
        basic_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#EAF0F8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 5)
        ]))
        story += [basic_table, Spacer(1, 0.12 * inch)]

        story.append(Paragraph("Molecular Structures", heading_style))
        img2d = save_2d_structure_png(mol)
        img3d = save_3d_structure_png(mol)
        if img2d:
            temp_images.append(img2d)
        if img3d:
            temp_images.append(img3d)

        if img2d and img3d:
            structure_table = Table([[
                Image(img2d, width=3.05 * inch, height=2.0 * inch),
                Image(img3d, width=3.05 * inch, height=2.0 * inch)
            ]], colWidths=[3.25 * inch, 3.25 * inch])
            structure_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER")
            ]))
            story += [structure_table, Spacer(1, 0.12 * inch)]
        elif img2d:
            story += [Image(img2d, width=4.8 * inch, height=3.2 * inch),
                      Spacer(1, 0.12 * inch)]

        story.append(Paragraph("Molecular Properties and Interpretation", heading_style))
        prop_data = [[
            Paragraph("<b>Property</b>", header_style),
            Paragraph("<b>Value</b>", header_style),
            Paragraph("<b>Interpretation</b>", header_style)
        ]]
        for key, value in properties.items():
            prop_data.append([
                Paragraph(str(key), small_style),
                Paragraph(str(value), small_style),
                Paragraph(get_property_interpretation(key, value, properties), small_style)
            ])
        prop_table = Table(
            prop_data, colWidths=[1.55 * inch, 0.9 * inch, 4.0 * inch], repeatRows=1
        )
        prop_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 3)
        ]))
        story += [prop_table, Spacer(1, 0.12 * inch)]

        story.append(Paragraph("Key Interpretation", heading_style))
        for point in build_key_interpretation_points(properties):
            story.append(Paragraph(point, body_style))
            story.append(Spacer(1, 0.04 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Python Libraries Used and Their Importance", heading_style))
    library_data = [
        [Paragraph("<b>Library</b>", header_style),
         Paragraph("<b>Importance in this project</b>", header_style)],
        [Paragraph("Streamlit", small_style),
         Paragraph("Creates the interactive web app, input controls and PDF download interface.", small_style)],
        [Paragraph("RDKit", small_style),
         Paragraph("Parses SMILES, validates structures, creates molecular drawings and 3D conformers, and calculates molecular descriptors.", small_style)],
        [Paragraph("Pandas", small_style),
         Paragraph("Organizes molecular properties in tables and supports CSV export.", small_style)],
        [Paragraph("Matplotlib and NumPy", small_style),
         Paragraph("Process molecular coordinates and generate a clear 2D projection of 3D geometry for the report.", small_style)],
        [Paragraph("py3Dmol", small_style),
         Paragraph("Displays interactive 3D molecular structures in the web application.", small_style)],
        [Paragraph("ReportLab", small_style),
         Paragraph("Builds the downloadable PDF with tables, figures and explanatory text.", small_style)],
        [Paragraph("io, os, tempfile", small_style),
         Paragraph("Handle in-memory PDF data and temporary image files.", small_style)]
    ]
    lib_table = Table(library_data, colWidths=[1.8 * inch, 4.9 * inch], repeatRows=1)
    lib_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5)
    ]))
    story += [lib_table, Spacer(1, 0.15 * inch)]

    story.append(Paragraph("Illustrative Python Code", heading_style))
    code_style = ParagraphStyle(
        "MultiReportCode", parent=styles["Code"], fontName="Courier",
        fontSize=7, leading=8.5, leftIndent=4, rightIndent=4
    )
    code_example = "\n".join([
        "from rdkit import Chem",
        "from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors",
        "",
        'smiles_list = ["CCO", "c1ccccc1"]  # Add more SMILES strings here',
        "",
        "for smiles in smiles_list:",
        "    mol = Chem.MolFromSmiles(smiles)",
        "    if mol is None:",
        '        print("Invalid SMILES:", smiles)',
        "        continue",
        "",
        '    print("SMILES:", smiles)',
        '    print("Formula:", rdMolDescriptors.CalcMolFormula(mol))',
        '    print("Molecular weight:", Descriptors.MolWt(mol))',
        '    print("LogP:", Crippen.MolLogP(mol))'
    ])
    story.append(Preformatted(code_example, code_style, maxLineLength=92))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(
        "This compact code example demonstrates processing multiple SMILES in one run. "
        "The complete application includes additional code for visualization, descriptor "
        "interpretation, student information and PDF generation. Calculated descriptors "
        "are computational results and should be interpreted appropriately.",
        body_style
    ))

    try:
        doc.build(story)
        return buffer.getvalue()
    finally:
        buffer.close()
        for image_path in temp_images:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception:
                pass


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧪 Virtual Lab")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "📖 Theory",

        "🧬 Molecular Visualization",

        "📊 Molecular Properties Report",

        "📈 Structure–Property Analysis",

        "📝 Assessment",

        "📄 Final Project Report"

    ]
)

st.sidebar.divider()

st.sidebar.markdown("### 👩‍🎓 Student Information")
student_name = st.sidebar.text_input("Student Name", key="student_name")
registration_number = st.sidebar.text_input("Registration Number", key="registration_number")

st.sidebar.markdown("""
### 🎓 Learning Objectives

Students can learn:

- SMILES notation
- Molecular representation
- 2D visualization
- 3D visualization
- Molecular descriptors
- Structure-property relationships
""")


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    # ========================================================
    # INSTITUTIONAL HEADER / COLLEGE LOGO
    # ========================================================
    logo_path = os.path.join(os.path.dirname(__file__), "srm_logo.jpeg")

    if os.path.exists(logo_path):
        logo_col1, logo_col2, logo_col3 = st.columns([1, 4, 1])
        with logo_col2:
            st.image(logo_path, use_container_width=True)

    st.markdown(
        '<div class="main-title">🧪 Cheminformatics Virtual Laboratory</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align:center; font-size:24px; font-weight:700; color:#1f4e79;">
        SRM Institute of Science and Technology, Tiruchirappalli
        </div>
        <div style="text-align:center; font-size:18px; font-weight:600; margin-top:5px;">
        Faculty of Engineering and Technology | School of Sciences | Division of Chemistry
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("## Welcome to the Virtual Laboratory")

    st.write("""
    This interactive virtual laboratory helps students understand
    important concepts in cheminformatics and computational chemistry.
    """)

    st.markdown("### 🧪 What can students do?")

    st.markdown("""
    ✅ Enter any SMILES notation

    ✅ Validate molecular structures

    ✅ Generate 2D structures

    ✅ Generate interactive 3D structures

    ✅ Select different molecular visualization styles

    ✅ Calculate complete molecular properties

    ✅ Download molecular reports

    ✅ Compare multiple molecules

    ✅ Complete an assessment
    """)

    st.markdown("### Example Molecules")

    example_df = pd.DataFrame({

        "Molecule": [

            "Ethanol",

            "Benzene",

            "Phenol",

            "Acetic Acid",

            "Aspirin",

            "Caffeine"
        ],

        "SMILES": [

            "CCO",

            "c1ccccc1",

            "Oc1ccccc1",

            "CC(=O)O",

            "CC(=O)Oc1ccccc1C(=O)O",

            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        ]
    })

    st.dataframe(
        example_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### 👨‍🏫 Developed by")
    st.markdown("**Dr. R. Subramanian**")
    st.markdown("Assistant Professor  ")
    st.markdown("Division of Chemistry, School of Sciences  ")
    st.markdown("Faculty of Engineering and Technology  ")
    st.markdown("SRM Institute of Science and Technology, Tiruchirappalli")

    st.info("📧 **For queries, contact:** rsmani84@gmail.com")


# ============================================================
# THEORY PAGE
# ============================================================

elif page == "📖 Theory":

    st.title("📖 Introduction to Cheminformatics")

    st.markdown("""
## What is Cheminformatics?

Cheminformatics combines:

- Chemistry ⚗️
- Computer Science 💻
- Data Analysis 📊

It uses computational techniques to store, process,
visualize and analyze chemical information.
""")

    st.markdown("""
## SMILES

**SMILES** means:

### Simplified Molecular Input Line Entry System

It represents a molecular structure using a text string.

Examples:

| Molecule | SMILES |
|---|---|
| Ethanol | CCO |
| Benzene | c1ccccc1 |
| Acetic Acid | CC(=O)O |
""")

    st.markdown("""
## Molecular Descriptors

Molecular descriptors provide numerical information
about a chemical structure.

Examples include:

- Molecular Weight
- LogP
- TPSA
- Hydrogen Bond Donors
- Hydrogen Bond Acceptors
- Rotatable Bonds
""")

    st.markdown("""
## 3D Molecular Models

Different visualization models help students understand
the molecular geometry.

### 🟢 Ball and Stick

Shows atoms as spheres and bonds as sticks.

### 🟡 Stick Model

Emphasizes chemical bonding.

### 🔵 Space Filling

Shows the approximate molecular volume.

### ⚫ Wireframe

Provides a simplified representation of the molecule.
""")


# ============================================================
# MOLECULAR VISUALIZATION PAGE
# ============================================================

elif page == "🧬 Molecular Visualization":

    st.title("🧬 Molecular Visualization Laboratory")

    st.info("Enter a new SMILES or load an example. The selected molecule will be used across the laboratory pages.")

    col1, col2 = st.columns([2, 1])

    with col1:
        smiles = st.text_input(
            "Enter SMILES",
            key="active_smiles"
        )

    with col2:
        st.selectbox(
            "Choose Example Molecule",
            list(EXAMPLE_MOLECULES.keys()),
            key="selected_example"
        )

        st.button(
            "Load Example",
            on_click=load_shared_example,
            key="load_visualization_example"
        )

    # Keep the latest valid molecule available for the other pages.
    mol = get_molecule(smiles)

    if mol is None:

        st.error("❌ Invalid SMILES notation.")

    else:

        st.success("✅ Valid Molecular Structure!")
        st.session_state.latest_smiles = smiles
        st.session_state.latest_properties = calculate_properties(mol)

        col1, col2 = st.columns([1.2, 1])

        with col1:

            st.subheader("2D Molecular Structure")

            image = generate_2d_image(mol)

            st.image(image)

        with col2:

            st.subheader("Molecular Information")

            st.write(
                "**Input SMILES:**",
                smiles
            )

            st.write(
                "**Canonical SMILES:**",
                Chem.MolToSmiles(mol)
            )

            st.write(
                "**Formula:**",
                rdMolDescriptors.CalcMolFormula(mol)
            )

            bond_info = calculate_bond_information(mol)

            st.write(
                "**Heavy Atoms:**",
                mol.GetNumHeavyAtoms()
            )

            st.write(
                "**Total Atoms (including H):**",
                bond_info["Total Atoms Including H"]
            )

            st.write(
                "**Heavy-Atom Bonds:**",
                bond_info["Heavy Atom Bonds"]
            )

            st.write(
                "**Sigma (σ) Bonds:**",
                bond_info["Sigma Bonds"]
            )

            st.write(
                "**Pi (π) Bonds:**",
                bond_info["Pi Bonds"]
            )

            st.write(
                "**Total Bonds (σ + π):**",
                bond_info["Total Bonds"]
            )

        st.divider()

        st.subheader("🌐 Interactive 3D Molecular Structure")

        visualization_style = st.selectbox(

            "Select Molecular Model",

            [

                "Ball and Stick",

                "Stick",

                "Space Filling",

                "Wireframe",

                "Cartoon"
            ]
        )

        show_surface = st.checkbox(
            "Show van der Waals Surface"
        )

        if st.button("🚀 Generate 3D Structure"):

            with st.spinner(
                "Generating 3D molecular structure..."
            ):

                mol3d = generate_3d_molecule(mol)

            if mol3d is None:

                st.error(
                    "Unable to generate the 3D structure."
                )

            else:

                display_3d_structure(

                    mol3d,

                    visualization_style,

                    show_surface
                )


# ============================================================
# COMPLETE MOLECULAR PROPERTIES REPORT
# ============================================================

elif page == "📊 Molecular Properties Report":

    st.title("📊 Complete Molecular Properties Report")

    st.info("""
Enter any valid SMILES notation. The system will automatically
generate a complete molecular properties report.
""")

    st.write("### Select or Enter Your Molecule")

    col1, col2 = st.columns([2, 1])

    with col1:
        smiles = st.text_input(
            "Enter New SMILES",
            key="active_smiles"
        )

    with col2:
        st.selectbox(
            "Choose Example Molecule",
            list(EXAMPLE_MOLECULES.keys()),
            key="selected_example"
        )

        st.button(
            "Load Example",
            on_click=load_shared_example,
            key="load_properties_example"
        )

    st.caption("Once you enter a valid SMILES or load an example, the complete 2D/3D structures and molecular properties below are generated for the same molecule.")

    if smiles:

        mol = get_molecule(smiles)

        if mol is None:

            st.error(
                "❌ Invalid SMILES notation. Please check the structure."
            )

        else:

            st.success(
                "✅ Valid Molecular Structure Detected!"
            )

            properties = calculate_properties(mol)

            canonical_smiles = Chem.MolToSmiles(mol)

            # Save the latest student-entered molecule for the Final Project Report
            st.session_state.latest_smiles = smiles
            st.session_state.latest_properties = properties

            # ------------------------------------------------
            # BASIC INFORMATION
            # ------------------------------------------------

            col1, col2 = st.columns([1.2, 1])

            with col1:

                st.subheader("🧬 2D Molecular Structure")

                image = generate_2d_image(mol)

                st.image(image)

            with col2:

                st.subheader("📋 Basic Information")

                st.write(
                    "**Input SMILES:**",
                    smiles
                )

                st.write(
                    "**Canonical SMILES:**",
                    canonical_smiles
                )

                st.write(
                    "**Molecular Formula:**",
                    properties["Molecular Formula"]
                )

                st.write(
                    "**Atoms:**",
                    properties["Number of Atoms"]
                )

                st.write(
                    "**Heavy Atoms:**",
                    properties["Heavy Atoms"]
                )

                st.write(
                    "**Heavy-Atom Bonds:**",
                    properties["Heavy Atom Bonds"]
                )

                st.write(
                    "**Sigma (σ) Bonds:**",
                    properties["Sigma (σ) Bonds"]
                )

                st.write(
                    "**Pi (π) Bonds:**",
                    properties["Pi (π) Bonds"]
                )

                st.write(
                    "**Total Bonds (σ + π):**",
                    properties["Total Bonds (σ + π)"]
                )

            st.divider()

            # ------------------------------------------------
            # PROPERTY METRICS
            # ------------------------------------------------

            st.subheader("📊 Important Molecular Properties")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Molecular Weight",
                f"{properties['Molecular Weight']} g/mol"
            )

            col2.metric(
                "LogP",
                properties["LogP"]
            )

            col3.metric(
                "TPSA",
                f"{properties['TPSA']} Å²"
            )

            col4.metric(
                "Ring Count",
                properties["Ring Count"]
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "H-Bond Donors",
                properties["Hydrogen Bond Donors"]
            )

            col2.metric(
                "H-Bond Acceptors",
                properties["Hydrogen Bond Acceptors"]
            )

            col3.metric(
                "Rotatable Bonds",
                properties["Rotatable Bonds"]
            )

            col4.metric(
                "Fraction Csp3",
                properties["Fraction Csp3"]
            )

            st.divider()

            # ------------------------------------------------
            # COMPLETE TABLE
            # ------------------------------------------------

            st.subheader("📑 Complete Molecular Property Table")

            report_data = [

                ["Input SMILES", smiles],

                ["Canonical SMILES", canonical_smiles]
            ]

            for key, value in properties.items():

                report_data.append(
                    [key, value]
                )

            report_df = pd.DataFrame(

                report_data,

                columns=[
                    "Property",
                    "Value"
                ]
            )

            st.dataframe(

                report_df,

                use_container_width=True,

                hide_index=True
            )

            # ------------------------------------------------
            # DOWNLOAD CSV
            # ------------------------------------------------

            csv = report_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(

                "⬇️ Download Properties Report (CSV)",

                data=csv,

                file_name="Molecular_Properties_Report.csv",

                mime="text/csv"
            )

            # ------------------------------------------------
            # DOWNLOAD PDF
            # ------------------------------------------------

            if st.button(
                "📄 Generate Molecular PDF Report"
            ):

                with st.spinner(
                    "Generating molecular PDF report..."
                ):

                    pdf = generate_molecule_pdf(

                        smiles,

                        mol,

                        properties
                    )

                st.download_button(

                    "⬇️ Download Molecular Properties PDF",

                    data=pdf,

                    file_name="Molecular_Properties_Report.pdf",

                    mime="application/pdf"
                )

            st.divider()

            # ------------------------------------------------
            # 3D VISUALIZATION
            # ------------------------------------------------

            st.subheader("🌐 3D Molecular Visualization")

            model_style = st.selectbox(

                "Select 3D Model",

                [

                    "Ball and Stick",

                    "Stick",

                    "Space Filling",

                    "Wireframe",

                    "Cartoon"
                ],

                key="properties_3d_style"
            )

            surface = st.checkbox(

                "Show van der Waals Surface",

                key="properties_surface"
            )

            if st.button(
                "🚀 Generate 3D Molecular Model",
                key="properties_generate_3d"
            ):

                with st.spinner(
                    "Generating 3D molecular model..."
                ):

                    mol3d = generate_3d_molecule(mol)

                if mol3d is None:

                    st.error(
                        "Unable to generate the 3D structure."
                    )

                else:

                    display_3d_structure(

                        mol3d,

                        model_style,

                        surface
                    )


# ============================================================
# STRUCTURE PROPERTY ANALYSIS
# ============================================================

elif page == "📈 Structure–Property Analysis":

    st.title("📈 Structure–Property Analysis")

    st.write("""
Compare the properties of multiple molecules.
Enter one molecule per line in the format:

**Molecule Name, SMILES**
""")

    default_data = """Ethanol,CCO
Benzene,c1ccccc1
Phenol,Oc1ccccc1
Acetic Acid,CC(=O)O
Aspirin,CC(=O)Oc1ccccc1C(=O)O"""

    molecule_input = st.text_area(

        "Enter Molecules",

        value=default_data,

        height=200
    )

    rows = []

    for line in molecule_input.splitlines():

        if "," not in line:

            continue

        name, smiles = line.split(",", 1)

        mol = get_molecule(smiles)

        if mol:

            prop = calculate_properties(mol)

            rows.append({

                "Molecule": name.strip(),

                "MW": prop["Molecular Weight"],

                "LogP": prop["LogP"],

                "TPSA": prop["TPSA"],

                "HBD": prop["Hydrogen Bond Donors"],

                "HBA": prop["Hydrogen Bond Acceptors"],

                "Rotatable Bonds": prop["Rotatable Bonds"],

                "Ring Count": prop["Ring Count"]
            })

    if rows:

        df = pd.DataFrame(rows)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("📋 Comparison Table")

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Comparison Table (CSV)",
            data=csv_data,
            file_name="structure_property_comparison.csv",
            mime="text/csv",
            key="download_structure_property_csv"
        )

    else:
        st.warning("Please enter at least one valid molecule in the format: Molecule Name, SMILES")


# ============================================================
# ASSESSMENT
# ============================================================

elif page == "📝 Assessment":

    st.title("📝 Cheminformatics Assessment")

    questions = [
        {"question":"What is the full form of SMILES?", "options":["Simplified Molecular Input Line Entry System","Standard Molecular Information Language Encoding System","Simple Molecular Identification and Labeling System"], "answer":"Simplified Molecular Input Line Entry System"},
        {"question":"What does a SMILES string represent?", "options":["A molecular structure using text","A spectroscopy technique","A laboratory instrument"], "answer":"A molecular structure using text"},
        {"question":"Which SMILES represents ethanol?", "options":["CCO","c1ccccc1","CC(=O)O"], "answer":"CCO"},
        {"question":"Which SMILES represents benzene?", "options":["c1ccccc1","CCO","O=C=O"], "answer":"c1ccccc1"},
        {"question":"In SMILES, parentheses are mainly used to indicate:", "options":["Branches","Molecular weight","Atom colour"], "answer":"Branches"},
        {"question":"Which property is commonly associated with molecular lipophilicity?", "options":["LogP","TPSA","HBD"], "answer":"LogP"},
        {"question":"What does TPSA describe?", "options":["Topological Polar Surface Area","Total Pi Surface Area","Thermal Property Surface Analysis"], "answer":"Topological Polar Surface Area"},
        {"question":"What does HBA mean?", "options":["Hydrogen Bond Acceptor","Heavy Bond Atom","Hydrogen Bond Analysis"], "answer":"Hydrogen Bond Acceptor"},
        {"question":"What does HBD mean?", "options":["Hydrogen Bond Donor","Heavy Bond Descriptor","Hydrogen Bond Distance"], "answer":"Hydrogen Bond Donor"},
        {"question":"A double bond contains how many pi (π) bonds?", "options":["1","2","0"], "answer":"1"},
        {"question":"A triple bond contains how many pi (π) bonds?", "options":["2","1","3"], "answer":"2"},
        {"question":"Which model is useful for visualizing approximate molecular volume?", "options":["Space Filling","Wireframe only","Text SMILES"], "answer":"Space Filling"}
    ]

    student_answers = []

    for i, q in enumerate(questions):

        answer = st.radio(
            q["question"],
            q["options"],
            index=None,
            key=f"assessment_v2_{i}"
        )

        student_answers.append(answer)

    if st.button("Submit Assessment"):

        if any(answer is None for answer in student_answers):
            st.warning("Please answer all questions before submitting the assessment.")
            st.stop()

        score = 0

        for i, q in enumerate(questions):

            if student_answers[i] == q["answer"]:
                score += 1

        percentage = (score / len(questions)) * 100

        st.success(
            f"Your Score: {score}/{len(questions)}"
        )

        st.metric(
            "Percentage",
            f"{percentage:.1f}%"
        )

        if percentage >= 80:

            st.balloons()

            st.success("🎉 Excellent performance!")

        elif percentage >= 50:

            st.info("👍 Good! Continue practicing.")

        else:

            st.warning(
                "📚 Please review the Theory section and try again."
            )

# ============================================================
# FINAL PROJECT REPORT
# ============================================================

elif page == "📄 Final Project Report":

    st.title("📄 Final Student Molecular Analysis Report")
    st.write(
        "Enter one SMILES string per line. The report will include every valid molecule "
        "in a single downloadable PDF."
    )

    name = st.text_input(
        "Student Name for Report",
        value=st.session_state.get("student_name", ""),
        key="final_report_name"
    )
    reg = st.text_input(
        "Registration Number for Report",
        value=st.session_state.get("registration_number", ""),
        key="final_report_reg"
    )

    default_smiles = st.session_state.get(
        "latest_smiles",
        st.session_state.get("active_smiles", "CCO")
    )
    multi_smiles_text = st.text_area(
        "Enter SMILES (one molecule per line)",
        value=st.session_state.get("final_report_smiles_multi", default_smiles),
        height=150,
        key="final_report_smiles_multi",
        help="Example: CCO on the first line, c1ccccc1 on the second line, and so on."
    )

    smiles_entries = [
        line.strip() for line in multi_smiles_text.splitlines() if line.strip()
    ]
    molecule_records = []
    invalid_smiles = []

    for smiles_item in smiles_entries:
        mol_item = get_molecule(smiles_item)
        if mol_item is None:
            invalid_smiles.append(smiles_item)
            continue
        props_item = calculate_properties(mol_item)
        molecule_records.append((smiles_item, mol_item, props_item))

    if invalid_smiles:
        st.warning(
            "These SMILES could not be interpreted and will not be included: "
            + ", ".join(invalid_smiles)
        )

    if molecule_records:
        st.success(f"{len(molecule_records)} valid molecule(s) ready for the report.")
        preview_rows = []
        for idx, (smiles_item, mol_item, props_item) in enumerate(molecule_records, start=1):
            preview_rows.append({
                "No.": idx,
                "Input SMILES": smiles_item,
                "Molecular Formula": props_item["Molecular Formula"],
                "Molecular Weight (g/mol)": props_item["Molecular Weight"]
            })
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

        if st.button("📄 Generate Single PDF for All Molecules"):
            with st.spinner(
                f"Generating one PDF report for {len(molecule_records)} molecule(s)..."
            ):
                pdf = generate_multi_final_report(name, reg, molecule_records)
            st.success("✅ Combined molecular report generated successfully!")
            st.download_button(
                label="⬇️ Download Combined Molecular Report PDF",
                data=pdf,
                file_name="Cheminformatics_Multi_Molecule_Report.pdf",
                mime="application/pdf",
                key="download_multi_molecule_pdf"
            )
    else:
        st.info("Enter at least one valid SMILES string to generate the report.")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🧪 Cheminformatics Virtual Laboratory | "
    "Developed using Python, Streamlit and RDKit"
)
