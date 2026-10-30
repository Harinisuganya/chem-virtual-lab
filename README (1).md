# SRM Cheminformatics Virtual Laboratory

An upgraded Streamlit educational app for molecular lookup, 2D/3D visualization, descriptor exploration, molecule comparison, quizzes, and downloadable reports.

## Project contributors

- **Harini M** — First Year, Cyber A — `RA2611030050013`
- **Harini Sri** — First Year, Cyber A — `RA2611030050010`

## Main features

- SRM Tiruchirappalli logo and project team details
- SMILES lookup through PubChem
- 2D structure image and available 3D conformer viewer
- Molecular formula, molecular weight, exact mass, XLogP, TPSA, hydrogen-bond counts, rotatable bonds and other available PubChem properties
- Basic bond information parsed from PubChem SDF records when available
- Molecule comparison table and descriptor chart
- Interactive quiz with score and feedback
- CSV and PDF report downloads

## Important technical note

This version intentionally **does not use RDKit**. It retrieves existing compound records from the PubChem PUG REST API. Therefore, an internet connection is required, and descriptor/3D availability depends on the PubChem record. It cannot calculate descriptors for an arbitrary structure that has no matching PubChem record. Bond counts parsed from SDF are estimates and are labelled accordingly.

## Run locally

1. Install Python 3.10 or newer.
2. Keep `app.py`, `requirements.txt`, `README.md`, and `srm_logo.jpeg` in the same folder.
3. Open a terminal in that folder and run:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload these files:
   - `app.py`
   - `requirements.txt`
   - `srm_logo.jpeg`
   - `README.md`
2. Open Streamlit Community Cloud and connect the repository.
3. Choose `app.py` as the app's main file.
4. Deploy. No `packages.txt` system packages are required for this version.

## Troubleshooting

- **Logo missing:** confirm `srm_logo.jpeg` is in the repository root beside `app.py`.
- **No matching compound:** verify the SMILES string or try one of the built-in examples. PubChem lookup is an exact structure search.
- **No 3D model:** some PubChem records do not provide a 3D conformer.
- **Network/API issue:** try again later; PubChem may temporarily limit or delay requests.
