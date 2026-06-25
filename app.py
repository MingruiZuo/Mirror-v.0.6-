import streamlit as st
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.structure_parser.pdb_parser import read_pdb, write_pdb
from modules.structure_parser.cif_parser import read_cif, write_cif
from modules.operation.mirror import mirror_structure
from modules.operation.cpa import pdb_to_coords, align_cpa, coords_to_atoms
from modules.visualization.viewer import show_structure
from modules.visualization.cif_viewer import render_cif_view
from modules.analysis.space_group import analyze_space_group
from modules.analysis.channel_map import channel_map_pipeline
from modules.analysis.pore_detection import pore_detection_pipeline, fast_pore_detection_pipeline
from modules.analysis.guest_orientation import orientation_pipeline
from modules.analysis.channel_accessibility import accessibility_pipeline
from modules.analysis.molecule_analysis import analyze_pdb


# ────────────────────────────────────────────
#  CoRE-COF database path
# ────────────────────────────────────────────
DB_DIR = Path(__file__).resolve().parent / "modules" / "database" / "CoRE-COFs_1242-v7.0"
XLSX_PATH = DB_DIR / "ALL-COF-1242.xlsx"


@st.cache_data(show_spinner="Loading CoRE-COF database…")
def load_cof_database():
    """Load the ALL-COF-1242.xlsx into a DataFrame and add a CIF path column."""
    df = pd.read_excel(XLSX_PATH)
    df["CIF Path"] = df["Number"].apply(lambda n: str(DB_DIR / f"{n}.cif"))
    return df


st.set_page_config(page_title="Chiral Toolkit")

st.title("Mirror (Demo v.0.6)")

# ── Homepage banner ──
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # allow large TIFFs
_banner_path = Path(r"D:\Mingrui_Zuo\Projects\PhD_Year2\APR-June\Figures\Big_Picture.tif")
if _banner_path.exists():
    try:
        _img = Image.open(_banner_path)
        st.image(_img, use_column_width=True)
    except Exception as e:
        st.caption(f"Banner not loaded: {e}")


# CREATE TABs
tab1, tab2, tab3, tab4, tab5, tab_infomation = st.tabs([
    "PDB Enantiomer",
    "CPA Alignment",
    "CIF Toolkits",
    "Accessible Orientations",
    "Database",
    "Information"
])

##################################################
# 1. PDB Enantiomer
##################################################
with tab1:

    st.header("Mirror Generator")
    st.write(
        "This tab performs a particular reflection operation. "
        "Supports **.pdb** and **.def** (RASPA) files."
    )
    with st.expander("Description"):

        st.write("""
        This module generates the mirror image of a molecule.

        The transformation is performed around the molecular centroid.

        Available planes:

        - YZ : invert x
        - XZ : invert y
        - XY : invert z

        ---
        **.def support** — RASPA molecule-definition files are
        parsed and converted to PDB for visualisation. A mirrored
        \*.def is also generated (only coordinates in the
        ``# atomic positions`` block are modified).
        """)

    uploaded_file1 = st.file_uploader(
        "Upload a PDB or DEF file",
        type=["pdb", "def"],
        key="mirror_upload"
    )

    if uploaded_file1 is not None:

        raw_text = uploaded_file1.getvalue().decode()
        is_def = uploaded_file1.name.lower().endswith(".def")

        if is_def:
            from modules.structure_parser.def_parser import (
                read_def, write_def, def_atoms_to_pdb,
            )
            atoms, def_lines = read_def(raw_text)
            pdb_text = def_atoms_to_pdb(atoms)
            st.success(
                f"{len(atoms)} atoms loaded from .def  |  "
                f"converted to PDB for preview"
            )
        else:
            atoms, other_lines = read_pdb(raw_text)
            pdb_text = raw_text
            st.success(f"{len(atoms)} atoms loaded.")

        original_name = uploaded_file1.name
        base_stem = Path(original_name).stem

        # ── Molecular Analysis ──
        mol = analyze_pdb(atoms)
        with st.expander("Molecular Analysis", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Formula:** {mol['composition_str']}")
                st.markdown(f"**Atoms:** {mol['n_atoms']}")
                st.markdown(f"**Total mass:** {mol['total_mass']:.2f} u")
                st.markdown(
                    f"**BBox (Å):** {mol['bbox_dims'][0]:.2f} × "
                    f"{mol['bbox_dims'][1]:.2f} × {mol['bbox_dims'][2]:.2f}"
                )
            with c2:
                vdw_min_e = mol['vdw_min_element']
                vdw_max_e = mol['vdw_max_element']
                st.markdown(
                    f"**VDW radius min:** {mol['vdw_min']:.2f} Å ({vdw_min_e})"
                )
                st.markdown(
                    f"**VDW radius max:** {mol['vdw_max']:.2f} Å ({vdw_max_e})"
                )
                cm = mol['centroid_mass']
                st.markdown(
                    f"**Mass-weighted centroid:** "
                    f"({cm[0]:.2f}, {cm[1]:.2f}, {cm[2]:.2f})"
                )
                st.markdown(
                    f"**Geometric centroid:** "
                    f"({mol['centroid_geom'][0]:.2f}, "
                    f"{mol['centroid_geom'][1]:.2f}, "
                    f"{mol['centroid_geom'][2]:.2f})"
                )

        axis = st.selectbox(
            "Mirror plane",
            ["YZ (invert x)", "XZ (invert y)", "XY (invert z)"],
            key="mirror_axis"
        )

        # ── Mirrored atoms (always computed) ──
        mirrored_atoms = mirror_structure(atoms, axis)

        # ── def → PDB download (always shown for .def) ──
        if is_def:
            def_to_pdb_name = base_stem + ".pdb"
            st.download_button(
                label="📄 Download as PDB (def → pdb)",
                data=pdb_text,
                file_name=def_to_pdb_name,
                mime="text/plain",
                key="dl_def2pdb",
            )

        # ── Mirrored def download (always shown for .def) ──
        if is_def:
            mirrored_def_text = write_def(def_lines, mirrored_atoms)
            mirrored_def_name = base_stem + "_mirrored.def"
            st.download_button(
                label="🔁 Download mirrored .def",
                data=mirrored_def_text,
                file_name=mirrored_def_name,
                mime="text/plain",
                key="dl_mirrored_def",
            )

        # ── Visualization ──
        if is_def:
            mirrored_pdb_text = def_atoms_to_pdb(mirrored_atoms)
        else:
            mirrored_pdb_text = write_pdb(mirrored_atoms, other_lines)

        col1, col2 = st.columns(2)

        with col1:

            card = st.container(border=True)

            with card:

                st.markdown("### Original Structure")

                show_structure(pdb_text)
                st.markdown("- the original molecule")

        
        with col2:

            card = st.container(border=True)

            with card:

                st.markdown("### Mirrored Structure")

                show_structure(mirrored_pdb_text)

                if is_def:
                    dl_label = "Download mirrored PDB (from .def)"
                    dl_name = base_stem + "_mirrored.pdb"
                else:
                    dl_label = "Download mirrored PDB"
                    dl_name = base_stem + "_mirrored.pdb"

                st.download_button(
                    label=dl_label,
                    data=mirrored_pdb_text,
                    file_name=dl_name,
                    mime="text/plain",
                    key="dl_mirrored"
                )
        



##################################################
# 2. CPA Alignment
##################################################
with tab2:

    st.header("CPA Alignment")
    st.write("This tab transforms molecule into a canonical principal axes (CPA) frame.")

    uploaded_file2 = st.file_uploader(
        "Upload a PDB file",
        type=["pdb"],
        key="cpa_upload"
    )


    if uploaded_file2 is not None:

        original_name = uploaded_file2.name
        new_name = Path(original_name).stem + "_cpa.pdb"
        new_name_m = Path(original_name).stem + "_mcpa.pdb"
        pdb_text = uploaded_file2.getvalue().decode()
        atoms, other_lines = read_pdb(pdb_text)
        st.success(f"{len(atoms)} atoms loaded.")

        # ── View controls (always visible) ──
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            show_h = st.checkbox("Show hydrogen atoms", value=True,
                                 key="tab2_show_h")
        with col_ctrl2:
            _style_map = {"stick": "stick", "sphere (CPK)": "sphere",
                          "line": "line"}
            _style_label = st.selectbox("Render style", list(_style_map.keys()),
                                        key="tab2_cpk")
            cpk_style = _style_map[_style_label]
        with col_ctrl3:
            method = st.selectbox(
                "Alignment method",
                ["mass-weighted"],
                key="alignment_method"
            )

        # ── Standardize button ──
        if st.button("Standardize Structure", key="standardize_button"):

            # Filter hydrogen atoms if needed
            work_atoms = atoms
            if not show_h:
                work_atoms = [a for a in atoms
                              if a["element"].upper() != "H"]

            st.info(f"Standardization method selected: {method}  |  "
                    f"Atoms used: {len(work_atoms)}/{len(atoms)}")

            # Molecular analysis
            mol = analyze_pdb(work_atoms)

            # Minimum required PLD = 2 × VDW envelope radius
            from modules.shared import VDW_RADII
            _ca = np.array([[a["x"], a["y"], a["z"]] for a in work_atoms])
            _el = [a["element"] for a in work_atoms]
            _rr = np.array([VDW_RADII.get(e.upper(), 1.70) for e in _el])
            _env_r = float((np.linalg.norm(_ca, axis=1) + _rr).max())
            min_pld = 2 * _env_r

            # CPA alignment
            coords, masses = pdb_to_coords(work_atoms)
            coords_cpa, R, com, info = align_cpa(coords, masses)
            cpa_atoms = coords_to_atoms(work_atoms, coords_cpa)
            output_text = write_pdb(cpa_atoms, other_lines)

            # Mirrored CPA
            mirrored_atoms = mirror_structure(work_atoms, axis="YZ (invert x)")
            coords_m, masses_m = pdb_to_coords(mirrored_atoms)
            coords_cpa_m, R_m, com_m, info_m = align_cpa(coords_m, masses_m)
            cpa_atoms_m = coords_to_atoms(mirrored_atoms, coords_cpa_m)
            output_text_m = write_pdb(cpa_atoms_m, other_lines)

            # ── Store in session state ──
            st.session_state["tab2_ok"] = True
            st.session_state["tab2_mol"] = mol
            st.session_state["tab2_min_pld"] = min_pld
            st.session_state["tab2_info"] = info
            st.session_state["tab2_info_m"] = info_m
            st.session_state["tab2_orig_pdb"] = pdb_text
            st.session_state["tab2_cpa_pdb"] = output_text
            st.session_state["tab2_mcpa_pdb"] = output_text_m
            st.session_state["tab2_mirrored_cart"] = write_pdb(mirrored_atoms,
                                                                other_lines)
            st.session_state["tab2_fname"] = new_name
            st.session_state["tab2_fname_m"] = new_name_m

            # Force re-run so display block picks up new state
            st.rerun()

        # ── Display results (if available) ──
        if st.session_state.get("tab2_ok", False):

            mol = st.session_state["tab2_mol"]
            min_pld = st.session_state["tab2_min_pld"]
            info = st.session_state["tab2_info"]
            info_m = st.session_state["tab2_info_m"]
            I1, I2, I3 = info["evals"]
            I_ratio = info["ratio"]
            I1m, I2m, I3m = info_m["evals"]
            I_rm = info_m["ratio"]

            # ── Molecular Analysis ──
            with st.expander("Molecular Analysis", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Formula:** {mol['composition_str']}")
                    st.markdown(f"**Atoms:** {mol['n_atoms']}")
                    st.markdown(f"**Total mass:** {mol['total_mass']:.2f} u")
                with c2:
                    bbox = mol["bbox_dims"]
                    st.markdown(
                        f"**BBox (Å):** {bbox[0]:.2f} × {bbox[1]:.2f} × {bbox[2]:.2f}"
                    )
                    st.markdown(f"**Min required PLD (Å):** {min_pld:.2f}")

            # ── Visualization ──
            col1, col2 = st.columns(2)

            with col1:
                card = st.container(border=True)
                with card:
                    st.markdown("### Original (Cartesian)")
                    show_structure(
                        st.session_state["tab2_orig_pdb"],
                        style=cpk_style, hide_hydrogen=not show_h,
                    )
                    st.markdown("- the original PDB")
                    deg_str = ""
                    if I_ratio < 0.05:
                        deg_str = "  ⚠️ Near-spherical (I₁/I₃ ≈ 0)"
                    elif I_ratio > 0.9:
                        deg_str = "  ⚠️ Degenerate (I₁ ≈ I₃)"
                    st.caption(
                        f"I₁ (min) = {I1:.1f}, I₂ = {I2:.1f}, "
                        f"I₃ (max) = {I3:.1f}  |  I₁/I₃ = {I_ratio:.3f}{deg_str}"
                    )

            with col2:
                card = st.container(border=True)
                with card:
                    st.markdown("### Mirrored (Cartesian)")
                    show_structure(
                        st.session_state["tab2_mirrored_cart"],
                        style=cpk_style, hide_hydrogen=not show_h,
                    )
                    st.markdown("- the mirrored PDB")
                    st.caption(
                        f"I₁ = {I1m:.1f}, I₂ = {I2m:.1f}, "
                        f"I₃ = {I3m:.1f}  |  I₁/I₃ = {I_rm:.3f}"
                    )

            col3, col4 = st.columns(2)
            with col3:
                card = st.container(border=True)
                with card:
                    st.markdown("### Original (CPA)")
                    show_structure(
                        st.session_state["tab2_cpa_pdb"],
                        style=cpk_style, hide_hydrogen=not show_h,
                    )
                    st.download_button(
                        label="Download PDB",
                        data=st.session_state["tab2_cpa_pdb"],
                        file_name=st.session_state["tab2_fname"],
                        mime="text/plain", key="dl_cpa",
                    )

            with col4:
                card = st.container(border=True)
                with card:
                    st.markdown("### Mirrored (CPA)")
                    show_structure(
                        st.session_state["tab2_mcpa_pdb"],
                        style=cpk_style, hide_hydrogen=not show_h,
                    )
                    st.download_button(
                        label="Download PDB",
                        data=st.session_state["tab2_mcpa_pdb"],
                        file_name=st.session_state["tab2_fname_m"],
                        mime="text/plain", key="dl_mcpa",
                    )

    with st.expander("Description"):

        md_path = Path(__file__).resolve().parent / "docs" / "CPA.md"
        md_text = md_path.read_text(encoding="utf-8")
        st.markdown(md_text, unsafe_allow_html=False)



with tab3:

    st.header("CIF Toolkits")

    # st.title("CIF Symmetry Analysis (COF-aware)")

    uploaded = st.file_uploader("Upload CIF", type=["cif"])

    if uploaded:

        text = uploaded.getvalue().decode()

        atoms, cell, _ = read_cif(text)

        st.success(f"{len(atoms)} atoms loaded")

        # =====================================================
        # STRUCTURE VISUALIZATION (NEW)
        # =====================================================
        st.subheader("Structure Visualization")
        # render_cif_view(atoms)
        col1, col2, col3 = st.columns(3)

        with col1:
            nx = st.slider("a direction", 1, 5, 1)

        with col2:
            ny = st.slider("b direction", 1, 5, 1)

        with col3:
            nz = st.slider("c direction", 1, 5, 1)

        show_cell = st.checkbox(
            "Show cell box",
            value=True
        )

        bg = st.selectbox("Background", ["white", "black"])

        render_cif_view(
            atoms,
            cell,
            nx=nx,
            ny=ny,
            nz=nz,
            show_cell=show_cell,
            background=bg
        )


        # =====================================================
        # symmetry analysis
        # =====================================================
        if st.button("Analyze symmetry"):

            result = analyze_space_group(atoms, cell)

            st.subheader(f"Space Group: {result['best_group']}")

            st.write( f"- No. {result['best_number']} (Confidence: {result['confidence']:.2f})")

            if result["confidence"] >= 0.8:
                st.success("High confidence")

            elif result["confidence"] >= 0.5:
                st.warning("Medium confidence")

            else:
                st.error("Low confidence")

            # =================================================
            # chirality
            # =================================================
            st.write("Sohncke:", result["is_sohncke"])
            st.write("Enantiomorphic:", result["is_enantiomorphic"])

            # =================================================
            # scan table
            # =================================================
            with st.expander("Symmetry Scan"):

                scan_data = []

                for prec, (symbol, number) in result["scan_results"].items():

                    scan_data.append({
                        "symprec": prec,
                        "space group": symbol,
                        "number": number
                    })

                st.dataframe(scan_data, use_container_width=True)

        # =====================================================
        # element-based channel map
        # =====================================================
        st.divider()
        st.subheader("Element-based Channel Map")

        with st.expander("What is this?"):
            st.write("""
            This tool detects 1D channels (pores) in a porous crystal by:
            1. Building a supercell (~100 Å in the ab-plane)
            2. Projecting all atoms onto the ab-plane → 2D Voronoi tessellation
            3. Identifying enclosed Voronoi vertices as pore centres
            4. Selecting the largest pore
            5. Projecting atoms around the pore wall onto a cylinder and flattening it

            The resulting map shows which elements line the pore wall.
            """)

        col_rmin, col_rexp = st.columns(2)
        with col_rmin:
            r_min = st.slider(
                "Min pore radius (Å)", min_value=1.0, max_value=10.0,
                value=3.0, step=0.5, key="r_min"
            )
        with col_rexp:
            expand_radius = st.slider(
                "Wall expand radius (Å)", min_value=3.0, max_value=20.0,
                value=12.0, step=1.0, key="expand_radius"
            )

        if st.button("Generate Channel Map", key="gen_channel_map"):
            with st.spinner("Building supercell and analysing pores…"):
                result = channel_map_pipeline(
                    atoms,
                    cell,
                    r_min=r_min,
                    expand_radius=expand_radius,
                    target_ab=100,
                    nc=4,
                )

            st.info(result["message"])

            if result["success"]:
                st.success(
                    f"Largest pore radius: **{result['pore_radius']:.2f} Å**  |  "
                    f"Supercell: {result['supercell_size'][0]}×{result['supercell_size'][1]}×{result['supercell_size'][2]}"
                )
                st.pyplot(result["fig"])
            else:
                st.warning(result["message"])
                if result["fig"] is not None:
                    st.pyplot(result["fig"])

        # =====================================================
        # grid-based pore detection & VDW heatmap
        # =====================================================
        st.divider()
        st.subheader("Grid-based Pore Detection & VDW Heatmap")

        with st.expander("How this differs from the Voronoi method above"):
            st.write("""
            This method uses an alternative **grid-based** approach:
            1. Build a supercell and project atoms onto the ab-plane
            2. Create a dense 2D grid and compute the distance from each grid point to the nearest atom
            3. Extract the **largest connected region** with distance ≥ probe radius
            4. Find the pore centre as the grid point **farthest from any atom**
            5. Generate a **VDW surface** point cloud (Fibonacci sphere per atom)
            6. Project VDW points onto a cylinder around the pore axis
            7. Create a **smooth Gaussian heatmap** (SPMS-like) of VDW intensity on the θ–z plane

            The heatmap reveals which wall regions are lined by which functional groups.
            """)

        col_sc, col_pr, col_gn = st.columns(3)
        with col_sc:
            sc_na = st.number_input("Supercell a", 1, 6, 3, key="pd_sc_a")
            sc_nb = st.number_input("Supercell b", 1, 6, 3, key="pd_sc_b")
            sc_nc = st.number_input("Supercell c", 1, 6, 3, key="pd_sc_c")
        with col_pr:
            probe_r = st.slider("Probe radius (Å)", 0.5, 5.0, 1.2, 0.1, key="pd_probe")
            vdw_n = st.slider("VDW points / atom", 10, 80, 50, 5, key="pd_vdw_n")
        with col_gn:
            grid_n = st.slider("Grid resolution", 100, 400, 200, 20, key="pd_grid_n")
            expand_val = st.slider(
                "Expand (Å) — 0 = auto minimum",
                0.0, 20.0, 0.0, 0.5, key="pd_expand",
                help="0 = auto-compute minimum expand for continuous closed ring",
            )

        if st.button("Run Pore Detection", key="run_pore_detect"):
            with st.spinner("Running grid-based pore detection…"):
                result = pore_detection_pipeline(
                    atoms, cell,
                    supercell=(sc_na, sc_nb, sc_nc),
                    probe_radius=probe_r,
                    grid_n=grid_n,
                    vdw_n_points=vdw_n,
                    expand_auto=(expand_val == 0.0),
                    expand_fixed=expand_val if expand_val > 0 else 5.0,
                )

            # Store for Turbo comparison
            st.session_state["last_grid_result"] = result
            st.session_state["_grid_just_ran"] = True

            st.info(result["message"])

            if result["success"]:

                # Run timing display
                t = result.get("timing", {})
                if t:
                    st.caption(
                        f"Runtime: {t.get('total', '?'):.2f}s total  |  "
                        f"pore detection {t.get('pore_detection', '?'):.2f}s  |  "
                        f"VDW {t.get('vdw_surface', '?'):.2f}s  |  "
                        f"figures {t.get('figures', '?'):.2f}s  |  "
                        f"{t.get('n_particles', '?')} projected particles  |  "
                        f"{t.get('ms_per_1k', '?'):.2f}ms/1k particles"
                    )

                st.caption("Distance Field & Pore Channel")
                st.pyplot(result["fig_channel"])

                st.caption("Cylindrical Projection (element-coloured)")
                st.pyplot(result["fig_cylinder"])

                st.caption("Smooth VDW Heatmap (SPMS-like)")
                st.pyplot(result["fig_heatmap"])

                # ── Theta periodicity / pore shape ──
                st.divider()
                st.subheader("Pore Shape Classification")
                pore_label = result.get("pore_label", "L0")
                n_peaks = result.get("n_peaks", 0)
                shape_desc = {
                    "L6": "Hexagonal pore — 6-fold symmetry",
                    "L4": "Square pore — 4-fold symmetry",
                    "L3": "Trigonal/clover pore — 3-fold symmetry",
                    "L2": "Diamond/elliptical pore — 2-fold symmetry",
                }.get(pore_label, "Irregular or unresolved pore shape")

                col_label, col_peaks = st.columns(2)
                with col_label:
                    st.metric("Pore Label", pore_label, delta=None)
                with col_peaks:
                    st.metric("Detected Walls", f"{n_peaks}", delta=None)
                st.caption(shape_desc)

                if result.get("fig_theta_proj"):
                    st.caption("θ-axis Projection (VDW intensity vs angle)")
                    st.pyplot(result["fig_theta_proj"])

                if result.get("theta_projection") is not None:
                    with st.expander("Angular profile data"):
                        theta_proj = result["theta_projection"]
                        peak_idx = result.get("peak_indices", [])
                        valley_idx = result.get("valley_indices", [])
                        st.write(f"Profile length: {len(theta_proj)} bins over [-π, π]")
                        if len(peak_idx) > 0:
                            deg_per_bin = 360.0 / len(theta_proj)
                            peak_deg = [round(i * deg_per_bin, 1) for i in peak_idx]
                            st.write(f"Peak θ positions (deg): {peak_deg}")
                        if len(valley_idx) > 0:
                            deg_per_bin = 360.0 / len(theta_proj)
                            valley_deg = [round(i * deg_per_bin, 1) for i in valley_idx]
                            st.write(f"Valley θ positions (deg): {valley_deg}")
            else:
                st.warning(result["message"])

        # =====================================================
        # Turbo mode — Voronoi-based ultra-fast pore detection
        # =====================================================
        st.divider()
        st.subheader("⚡ Turbo Mode — Ultra-Fast Pore Detection (Voronoi + KDTree)")

        with st.expander("How Turbo differs from the grid-based method above"):
            st.write("""
            **Turbo mode** skips the expensive distance field and VDW surface entirely:

            1. Build a **minimal supercell** (2×2×1 by default)
            2. **Voronoi tessellation** of atomic positions → pore centre = Voronoi vertex farthest from any atom
            3. **KDTree** lookup → pore radius + nearest-neighbour angular profile
            4. **FFT** on angular profile → pore label (L2/L3/L4/L6…)
            5. **Cylindrical projection** of raw atom positions → element-coloured wall map

            **Expected speedup: 50–500×** (from ~seconds to ~milliseconds).
            Results may differ slightly from the grid-based method — use side-by-side to compare.
            """)

        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            t_sc_a = st.number_input("Supercell a", 1, 4, 2, key="turbo_sc_a")
            t_sc_b = st.number_input("Supercell b", 1, 4, 2, key="turbo_sc_b")
            t_sc_c = st.number_input("Supercell c", 1, 3, 1, key="turbo_sc_c")
        with t_col2:
            t_expand = st.slider("Expand (Å)", 1.0, 20.0, 5.0, 0.5, key="turbo_expand")
            t_rays = st.slider("Angular bins (rays)", 18, 180, 72, 6, key="turbo_rays",
                               help="More bins = finer angular resolution for pore shape")
        with t_col3:
            st.caption("")
            st.caption("")
            st.caption("")
            run_turbo = st.button("⚡ Run Turbo", key="run_turbo", type="primary")

        if run_turbo:
            with st.spinner("Running turbo pore detection…"):
                t_result = fast_pore_detection_pipeline(
                    atoms, cell,
                    supercell=(t_sc_a, t_sc_b, t_sc_c),
                    expand=t_expand,
                    n_rays=t_rays,
                )

            if t_result["success"]:
                # Timing
                tt = t_result.get("timing", {})
                st.info(t_result["message"])

                # Comparison layout: side-by-side with the grid method (if available)
                if st.session_state.get("last_grid_result"):
                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        st.caption("**Grid-based (Precise)**")
                        gr = st.session_state["last_grid_result"]
                        if gr.get("fig_channel"):
                            st.pyplot(gr["fig_channel"])
                        st.metric("Pore radius (grid)", f"{gr.get('pore_radius', 0):.2f} Å")
                        st.caption(f"Runtime: {gr.get('timing', {}).get('total', '?'):.3f}s")
                    with comp_col2:
                        st.caption("**Turbo (Fast)**")
                        if t_result.get("fig_channel"):
                            st.pyplot(t_result["fig_channel"])
                        st.metric("Pore radius (turbo)", f"{t_result.get('pore_radius', 0):.2f} Å")
                        st.caption(f"Runtime: {t_result.get('timing', {}).get('total', '?'):.4f}s")
                    st.divider()

                # Timing details
                if tt:
                    st.caption(
                        f"⏱️ Build {tt.get('build_supercell', 0):.4f}s  |  "
                        f"Voronoi {tt.get('voronoi', 0):.4f}s  |  "
                        f"Profile+FFT {tt.get('profile_fft', 0):.4f}s  |  "
                        f"Projection {tt.get('cyl_projection', 0):.4f}s  |  "
                        f"Figures {tt.get('figures', 0):.4f}s  |  "
                        f"**Total {tt.get('total', 0):.4f}s**"
                    )

                # Results
                t_tab1, t_tab2, t_tab3 = st.tabs([
                    "Voronoi Map", "Cylindrical Projection", "Pore Shape"])

                with t_tab1:
                    if t_result.get("fig_channel"):
                        st.pyplot(t_result["fig_channel"])
                    st.caption("Red X = pore centre, dashed circle = pore radius, "
                               "blue dots = Voronoi vertices")

                with t_tab2:
                    if t_result.get("fig_cylinder"):
                        st.pyplot(t_result["fig_cylinder"])
                    else:
                        st.warning("No wall atoms found in cylindrical projection range.")
                    with st.expander("Wall element summary"):
                        cd = t_result.get("channel_data")
                        if cd and len(cd.get("elements", [])) > 0:
                            elems = cd["elements"]
                            from collections import Counter
                            elem_counts = Counter(elems)
                            st.write(dict(elem_counts))
                        else:
                            st.write("No data.")

                with t_tab3:
                    pore_label = t_result.get("pore_label", "L0")
                    n_peaks = t_result.get("n_peaks", 0)
                    shape_desc = {
                        "L6": "Hexagonal pore — 6-fold symmetry",
                        "L4": "Square pore — 4-fold symmetry",
                        "L3": "Trigonal/clover pore — 3-fold symmetry",
                        "L2": "Diamond/elliptical pore — 2-fold symmetry",
                    }.get(pore_label, "Irregular or unresolved pore shape")

                    col_tl, col_tp = st.columns(2)
                    with col_tl:
                        st.metric("Pore Label", pore_label)
                    with col_tp:
                        st.metric("Detected Walls", f"{n_peaks}")
                    st.caption(shape_desc)

                    if t_result.get("fig_theta_proj"):
                        st.caption("θ-axis Profile (wall distance vs angle)")
                        st.pyplot(t_result["fig_theta_proj"])

                    # Compare with grid-based classification if available
                    if st.session_state.get("last_grid_result"):
                        gr = st.session_state["last_grid_result"]
                        g_label = gr.get("pore_label", "?")
                        g_radius = gr.get("pore_radius", 0)
                        t_radius = t_result.get("pore_radius", 0)
                        diff = abs(t_radius - g_radius)
                        st.caption(
                            f"Grid: {g_label}, R={g_radius:.2f}Å  |  "
                            f"Turbo: {pore_label}, R={t_radius:.2f}Å  |  "
                            f"ΔR = {diff:.3f}Å"
                        )
            else:
                st.warning(t_result["message"])

        # Store the last grid result for comparison
        # (Set in the grid-based section above via session_state trick)
        # We add a hidden mechanism: after grid run, save to session_state
        if st.session_state.get("_grid_just_ran", False):
            st.session_state["last_grid_result"] = st.session_state["_grid_result"]
            st.session_state["_grid_just_ran"] = False

        if st.session_state.get("_turbo_just_ran", False):
            st.session_state["_turbo_just_ran"] = False


##################################################
# 4. Accessible Orientations
##################################################
with tab4:

    st.header("Accessible Orientations")
    st.write(
        "Place a guest PDB molecule inside a host CIF crystal and scan all "
        "orientations to find clash-free poses."
    )

    with st.expander("How it works"):
        st.write("""
        1. **Upload** a host CIF (porous framework) and a guest PDB (small molecule).
        2. The guest centroid is fixed at a chosen position relative to the host.
        3. The guest is rotated through (θ, φ) samples on the unit sphere.
        4. For each orientation, VDW clash detection determines if the pose is accessible.
        5. Results are shown as:
           - **3D sphere**: green = clash-free, red = collision
           - **Clearance sphere**: colour = minimum guest–host distance
           - **Mollweide map**: global view of accessible orientations
           - **2D heatmap**: θ–φ plane coloured by clearance
        """)

    col_host, col_guest = st.columns(2)

    with col_host:
        host_file = st.file_uploader(
            "Upload host CIF", type=["cif"], key="orient_host"
        )

    with col_guest:
        guest_file = st.file_uploader(
            "Upload guest PDB", type=["pdb"], key="orient_guest"
        )

    # Clear cache when files change
    if host_file is None or guest_file is None:
        for key in ["orient_results", "ch_cache", "std_host", "cpa_guest", "mcpa_guest"]:
            if key in st.session_state:
                del st.session_state[key]

    if host_file is not None and guest_file is not None:

        # Parse
        host_text = host_file.getvalue().decode()
        guest_text = guest_file.getvalue().decode()

        host_atoms, host_cell, _ = read_cif(host_text)
        guest_atoms, _ = read_pdb(guest_text)

        # ── 1. Standardise: CIF → [0,1) frac; PDB → CPA frame ──
        from modules.analysis.channel_accessibility import (
            standardize_cif, cpa_guest, mcpa_guest,
            guest_pdb_string, merge_preview_pdb, channel_to_cif_string,
            centralized_channel, accessibility_pipeline,
        )
        std_host = standardize_cif(host_atoms, host_cell)
        cpa_g = cpa_guest(guest_atoms)
        mcpa_g = mcpa_guest(guest_atoms)
        # Cache in session_state
        st.session_state["std_host"] = std_host
        st.session_state["cpa_guest"] = cpa_g
        st.session_state["mcpa_guest"] = mcpa_g

        st.success(
            f"Host: {len(std_host)} atoms (standardised)  |  "
            f"Guest: {len(guest_atoms)} atoms (CPA)"
        )

        # ── 2. 2×2 preview: original / mirrored / CPA / MCPA ──
        with st.expander("Guest structure preview", expanded=True):
            gcol_a, gcol_b = st.columns(2)
            with gcol_a:
                st.caption("Original")
                show_structure(write_pdb(guest_atoms, []), height=200, width=280)
            with gcol_b:
                st.caption("Mirrored")
                from modules.operation.mirror import mirror_structure
                mg = mirror_structure(guest_atoms, "YZ (invert x)")
                show_structure(write_pdb(mg, []), height=200, width=280)
            gcol_c, gcol_d = st.columns(2)
            with gcol_c:
                st.caption("CPA-guest")
                show_structure(guest_pdb_string(cpa_g), height=200, width=280)
            with gcol_d:
                st.caption("MCPA-guest")
                show_structure(guest_pdb_string(mcpa_g), height=200, width=280)

        # ── 3. Centralised channel (cached) ──
        # ── CPA toggle and collision method ──
        ccol1, ccol2 = st.columns(2)
        with ccol1:
            use_cpa = st.checkbox("CPA-align guest before scan", value=True,
                                  key="orient_cpa",
                                  help="Align principal axes to Cartesian axes first")
        with ccol2:
            collision_method = st.selectbox(
                "Collision method",
                ["voxel", "vdw"],
                key="orient_collision",
                help="voxel = 3D distance field; vdw = pairwise VDW radii",
            )

        # ── Channel extraction (button-triggered) ──
        st.divider()
        st.subheader("Channel Extraction")
        ch_col1, ch_col2, ch_col3 = st.columns(3)
        with ch_col1:
            _ch_expand = st.slider(
                "Expand (\u00c5) \u2014 0 = auto",
                0.0, 20.0, 0.0, 0.5, key="ch_expand",
                help="0 = auto-compute optimal expand",
            )
        with ch_col2:
            _ch_nc = st.number_input("Repeats along c", 1, 6, 1, key="ch_nc")
        with ch_col3:
            _ch_sa = st.number_input("Supercell ab", 1, 5, 2, key="ch_sa")

        if st.button("Extract Channel", key="run_channel"):
            with st.spinner("Extracting centralised channel\u2026"):
                from modules.analysis.channel_accessibility import centralized_channel
                ch_res = centralized_channel(
                    std_host, host_cell,
                    supercell_ab=_ch_sa, nc=_ch_nc, probe_radius=1.2,
                )
                st.session_state["ch_cache"] = ch_res

        if st.session_state.get("ch_cache", {}).get("success", False):
            ch_res = st.session_state["ch_cache"]
            ch_ats = ch_res["channel_atoms"]

            with st.expander("Channel Preview", expanded=True):
                combined = merge_preview_pdb(ch_ats, cpa_g if use_cpa else mcpa_g)
                show_structure(combined, height=450, width=650)
                st.caption(
                    f"Pore R={ch_res['pore_radius']:.2f}\u00c5, "
                    f"expand={ch_res['expand_radius']:.1f}\u00c5, "
                    f"nc={_ch_nc}, "
                    f"{len(ch_ats)} channel atoms + {len(cpa_g)} CPA-guest atoms"
                )
                dl1, dl2 = st.columns(2)
                cif_name = Path(host_file.name).stem if host_file else "channel"
                with dl1:
                    cif_str = channel_to_cif_string(
                        ch_ats, cif_name=f"{cif_name}_channel_{_ch_nc}",
                    )
                    st.download_button(
                        "Download channel CIF",
                        data=cif_str,
                        file_name=f"{cif_name}_channel_{_ch_nc}.cif",
                        mime="text/plain",
                        key="dl_channel_cif",
                    )
                with dl2:
                    st.download_button(
                        "Download complex PDB",
                        data=combined,
                        file_name=f"{cif_name}_complex_nc{_ch_nc}.pdb",
                        mime="text/plain",
                        key="dl_complex_pdb",
                    )

            acc_col1, acc_col2 = st.columns(2)
            with acc_col1:
                with st.expander("Fit Analysis", expanded=False):
                    from modules.analysis.molecule_analysis import analyze_pdb
                    gm = analyze_pdb(guest_atoms)
                    bbox = gm["bbox_dims"]
                    gd = (bbox[0]**2 + bbox[1]**2 + bbox[2]**2)**0.5
                    pd_ = 2.0 * ch_res["pore_radius"]
                    cl = pd_ - gd
                    st.markdown(f"**Pore diameter:** {pd_:.2f} \u00c5")
                    st.markdown(f"**Guest box diag.:** {gd:.2f} \u00c5")
                    st.markdown(f"**Guest box:** {bbox[0]:.2f}\u00d7{bbox[1]:.2f}\u00d7{bbox[2]:.2f} \u00c5")
                    if cl > 0:
                        st.success(f"Likely fits (clearance = {cl:.2f} \u00c5)")
                    elif cl > -2:
                        st.warning(f"Tight fit (clearance = {cl:.2f} \u00c5)")
                    else:
                        st.error(f"Unlikely to fit (clearance = {cl:.2f} \u00c5)")

            with acc_col2:
                with st.expander("Channel Accessibility", expanded=False):
                    cell_len = max(host_cell["a"], host_cell["b"])
                    ca_res = accessibility_pipeline(
                        ch_ats, cpa_g if use_cpa else mcpa_g,
                        cell_length=cell_len,
                        guest_position_z=0.0,
                        grid_n=60, clash_scale=0.75,
                    )
                    if ca_res["success"]:
                        st.info(ca_res["message"])
                        st.pyplot(ca_res["fig"])
                    else:
                        st.warning(ca_res["message"])

        # Parameters
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            theta_steps = st.number_input(
                "Theta steps (polar)", 5, 90, 37, key="orient_theta"
            )
            phi_steps = st.number_input(
                "Phi steps (azimuthal)", 8, 180, 72, key="orient_phi"
            )
        with col_p2:
            clash_scale = st.slider(
                "Clash scale", 0.5, 1.0, 0.75, 0.05, key="orient_clash"
            )
            mirror = st.selectbox(
                "Mirror plane",
                ["none", "xy", "xz", "yz"],
                key="orient_mirror",
            )
        with col_p3:
            use_periodic = st.checkbox(
                "Periodic boundaries", value=True, key="orient_periodic"
            )
            st.caption(
                "Guest position defaults to host centroid. "
                "To customise, enter coordinates below."
            )
            custom_pos = st.checkbox(
                "Custom guest position", value=False, key="orient_custom_pos"
            )

        # Default position = origin (centralised channel); custom if enabled
        if custom_pos:
            col_xyz = st.columns(3)
            with col_xyz[0]:
                pos_x = st.number_input("X (\u00c5)", value=0.0,
                                        key="orient_x", format="%.2f")
            with col_xyz[1]:
                pos_y = st.number_input("Y (\u00c5)", value=0.0,
                                        key="orient_y", format="%.2f")
            with col_xyz[2]:
                pos_z = st.number_input("Z (\u00c5)", value=0.0,
                                        key="orient_z", format="%.2f")
            guest_position = [pos_x, pos_y, pos_z]
        else:
            guest_position = [0.0, 0.0, 0.0]

        # Show current guest position
        st.caption(
            f"Guest centroid: ({guest_position[0]:.2f}, "
            f"{guest_position[1]:.2f}, {guest_position[2]:.2f}) \u00c5"
        )

        # ── Scan button ──
        if st.button("Scan Orientations", key="run_orient"):

            total_samples = theta_steps * phi_steps

            with st.spinner(f"Scanning {total_samples} orientations\u2026"):
                result = orientation_pipeline(
                    host_atoms, host_cell, guest_atoms,
                    theta_steps=theta_steps,
                    phi_steps=phi_steps,
                    clash_scale=clash_scale,
                    guest_position=guest_position,
                    mirror_plane=mirror,
                    use_periodic=use_periodic,
                    collision_method=collision_method,
                    use_cpa=use_cpa,
                )

            # Store in session state so it survives widget changes
            st.session_state["orient_results"] = result

        # ── Display persisted results (if available) ──
        if st.session_state.get("orient_results") is not None:
            result = st.session_state["orient_results"]

            # Check if the result is stale (files changed)
            result_ok = result.get("success", False)

            if result_ok:
                st.info(result["message"])

                frac = result["n_free"] / max(result["n_total"], 1) * 100
                st.metric(
                    "Clash-free orientations",
                    f"{result['n_free']} / {result['n_total']} ({frac:.1f}%)"
                )

                # Plots
                tab_h = st.tabs(["Clearance Heatmap"])[0]

                with tab_h:
                    if result["fig_heatmap"] is not None:
                        st.pyplot(result["fig_heatmap"])
                    else:
                        st.warning("Heatmap not available")

            else:
                st.error(result["message"])

        # ── Pore Interface: 3-region partition + orientation map ──
        st.divider()
        st.subheader("Pore Interface — Three-Region Partition")
        with st.expander("How this works"):
            st.write("""
            **Rolling-sphere probe** — The guest's minimum enclosing VDW
            sphere is rolled through the channel cross-section:

            - **Green** (free): envelope sphere fits — all orientations OK.
            - **Amber** (restricted): only centroid fits — limited orientations.
            - **Red** (blocked): centroid does not fit.
            """)

        from modules.analysis.guest_prep import cpa_align, generate_enantiomer
        from modules.analysis.channel_accessibility import guest_circumsphere_radius
        from modules.shared import VDW_RADII
        from modules.analysis.pore_interface import (
            three_region_partition, orientation_sphere_map, run_dual_orientation,
            plot_three_region, plot_orientation_sphere, plot_heatmap,
            plot_dual_comparison, plot_binary_heatmap, plot_dual_binary,
        )

        # Prepare guest once and cache
        if "guest_prep" not in st.session_state:
            gl = cpa_align(guest_atoms)
            gd = generate_enantiomer(guest_atoms)
            xyz_L = np.array([[a["x"], a["y"], a["z"]] for a in gl])
            elems = [str(a.get("element", "C")) for a in gl]
            envelope_r = guest_circumsphere_radius(xyz_L, elems)
            gp = {
                "guest_L": gl,
                "guest_D": gd,
                "envelope_r": envelope_r,
            }
            st.session_state["guest_prep"] = gp
        gp = st.session_state["guest_prep"]

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("Run Three-Region Partition", key="run_3region"):
                ch_data = st.session_state.get("ch_cache")
                if ch_data is None or not ch_data.get("success", False):
                    st.warning("Please run Channel Extraction first.")
                else:
                    with st.spinner("Computing three-region partition…"):
                        part = three_region_partition(
                            ch_data["channel_atoms"],
                            gp["guest_L"], gp["guest_D"],
                            gp["envelope_r"], grid_n=60,
                        )
                        st.session_state["three_region"] = part
                        reg = part["region"]
                        pct_full = (reg == 0).sum() / (reg >= 0).sum() * 100
                        pct_restr = (reg == 1).sum() / (reg >= 0).sum() * 100
                        pct_inacc = (reg == 2).sum() / (reg >= 0).sum() * 100
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Free", f"{pct_full:.1f}%")
                        c2.metric("Restricted", f"{pct_restr:.1f}%")
                        c3.metric("Blocked", f"{pct_inacc:.1f}%")

        # ── Interactive pore explorer ──
        part = st.session_state.get("three_region")
        if part is not None:
            st.subheader("Pore Explorer")
            st.caption("Move sliders to position crosshair, then Compute.")

            if "cx" not in st.session_state:
                st.session_state["cx"] = 0.0
            if "cy" not in st.session_state:
                st.session_state["cy"] = 0.0

            col_px, col_py, col_pz = st.columns([2, 2, 1])
            with col_px:
                cx = st.slider("x (A)", -10.0, 10.0,
                               st.session_state["cx"], 0.2, key="cx")
            with col_py:
                cy = st.slider("y (A)", -10.0, 10.0,
                               st.session_state["cy"], 0.2, key="cy")
            with col_pz:
                st.markdown("####"); st.markdown("#####")
                doit = st.button("Compute", key="run_os", use_container_width=True)

            st.pyplot(plot_three_region(part, highlight_pos=(cx, cy)))
            plt.close("all")

            if doit:
                with st.spinner("Computing orientation sphere…"):
                    dual = run_dual_orientation(part, cx, cy)
                    st.session_state["dual_res"] = dual

            dual = st.session_state.get("dual_res")
            if dual is not None:
                rL, rD = dual["res_L"], dual["res_D"]
                rx, ry = dual["pos"]

                # Region info
                ix = np.argmin(np.abs(part["X"][0] - rx))
                iy = np.argmin(np.abs(part["Y"][:, 0] - ry))
                reg = part["region"][iy, ix] if (0 <= ix < part["X"].shape[1]
                        and 0 <= iy < part["Y"].shape[0]) else -1
                labels = {-1: "outside pore", 0: "free",
                          1: "restricted", 2: "blocked"}
                st.info(f"({rx:.1f}, {ry:.1f}) → **{labels.get(reg, '?')}**")

                binary_mode = st.checkbox("Binary heatmap (free/blocked)",
                                          value=False, key="binary_hm")

                _hm = plot_binary_heatmap if binary_mode else plot_heatmap

                tab_l, tab_d, tab_delta = st.tabs(
                    ["L-enantiomer", "D-enantiomer", "L vs D"])
                with tab_l:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.pyplot(plot_orientation_sphere(rL))
                    with col_b:
                        st.pyplot(_hm(rL))
                    plt.close("all")
                with tab_d:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.pyplot(plot_orientation_sphere(rD))
                    with col_b:
                        st.pyplot(_hm(rD))
                    plt.close("all")
                with tab_delta:
                    if binary_mode:
                        st.pyplot(plot_dual_binary(dual))
                    else:
                        st.pyplot(plot_dual_comparison(dual))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("L free", f"{rL['n_free']}/{rL['n_total']}"
                              f" ({rL['frac_free']*100:.1f}%)")
                    c2.metric("D free", f"{rD['n_free']}/{rD['n_total']}"
                              f" ({rD['frac_free']*100:.1f}%)")
                    c3.metric("Delta",
                              f"{(rL['frac_free']-rD['frac_free'])*100:+.1f}%",
                              delta_color="inverse")
                    plt.close("all")

    else:
        st.info("Please upload both a host CIF and a guest PDB file to begin.")


##################################################
# 5. Database
##################################################
with tab5:

    st.header("CoRE-COF Database Browser")
    st.write("Browse, search, and analyse entries from the CoRE-COF 1242 database.")

    try:
        cof_df = load_cof_database()
    except Exception as e:
        st.error(f"Failed to load database: {e}")
        st.stop()

    # ── Search ──
    search = st.text_input("Search COF", placeholder="e.g. COF-300, ZIF-8, 18C6…")

    # Filter
    filtered = cof_df.copy()
    if search:
        mask = (
            filtered["Input name"].str.contains(search, case=False, na=False)
            | filtered["COF name"].str.contains(search, case=False, na=False)
            | filtered["Number"].astype(str).str.contains(search, na=False)
        )
        filtered = filtered[mask]

    st.caption(f"{len(filtered)} / {len(cof_df)} entries")

    # ── Display table ──
    display_cols = ["Number", "Input name", "Type", "Topology",
                    "PLD (Å)", "LCD (Å)", "Density (g/cm3)"]
    st.dataframe(
        filtered[display_cols].set_index("Number"),
        use_container_width=True,
        height=300,
    )

    # ── Select & analyse ──
    if len(filtered) > 0:
        selected_num = st.selectbox(
            "Select a COF to analyse",
            options=filtered["Number"].tolist(),
            format_func=lambda n: f"{n} — {filtered[filtered['Number']==n]['Input name'].values[0]}",
            key="db_selected",
        )

        row = filtered[filtered["Number"] == selected_num].iloc[0]
        cif_path = row["CIF Path"]

        # Show metadata
        with st.expander("Entry details", expanded=False):
            meta_cols = ["Input name", "COF name", "Type", "Topology",
                         "reference", "Density (g/cm3)", "PLD (Å)", "LCD (Å)",
                         "Sacc (m2/cm-3)", "Sacc (m2/g-1)", "φ", "Vf (cm3/g-1)"]
            for c in meta_cols:
                st.markdown(f"**{c}:** {row[c]}")

        # Import button
        if st.button("Import CIF & Run Pore Detection", key="db_import"):

            with st.spinner(f"Loading COF #{selected_num}…"):
                cif_text = Path(cif_path).read_text(encoding="utf-8")
                db_atoms, db_cell, _ = read_cif(cif_text)

            st.success(
                f"**{row['Input name']}** (#{selected_num}) — "
                f"{len(db_atoms)} atoms loaded"
            )

            # Store in session_state so other tabs can access
            st.session_state["db_cif_text"] = cif_text
            st.session_state["db_atoms"] = db_atoms
            st.session_state["db_cell"] = db_cell
            st.session_state["db_name"] = f"{selected_num}_{row['Input name']}"

            # Show structure
            st.subheader("Structure")
            render_cif_view(db_atoms, db_cell, nx=2, ny=2, nz=1,
                            show_cell=True, background="white")

            # Symmetry
            st.subheader("Symmetry Analysis")
            sym_result = analyze_space_group(db_atoms, db_cell)
            st.write(
                f"**Space Group:** {sym_result['best_group']} "
                f"(No. {sym_result['best_number']})  |  "
                f"Confidence: {sym_result['confidence']:.2f}  |  "
                f"Sohncke: {sym_result['is_sohncke']}  |  "
                f"Enantiomorphic: {sym_result['is_enantiomorphic']}"
            )

            # Pore detection (grid-based)
            st.subheader("Pore Detection (Grid-based)")
            with st.spinner("Running pore detection…"):
                pore_result = pore_detection_pipeline(
                    db_atoms, db_cell,
                    supercell=(3, 3, 1),
                    probe_radius=1.2,
                    grid_n=150,
                    vdw_n_points=50,
                )

            if pore_result["success"]:
                
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption("Distance Field & Pore Channel")
                    st.pyplot(pore_result["fig_channel"])
                with col_b:
                    st.caption("Cylindrical Projection")
                    st.pyplot(pore_result["fig_cylinder"])
                    st.caption("Smooth VDW Heatmap")
                    st.pyplot(pore_result["fig_heatmap"])
                st.info(pore_result["message"].replace("Å", "A"))

                # ── Pore shape classification ──
                st.subheader("Pore Shape")
                pore_label = pore_result.get("pore_label", "L0")
                n_peaks = pore_result.get("n_peaks", 0)
                shape_desc = {
                    "L6": "Hexagonal pore (6-fold symmetry)",
                    "L4": "Square pore (4-fold symmetry)",
                    "L3": "Trigonal/clover pore (3-fold symmetry)",
                    "L2": "Diamond/elliptical pore (2-fold symmetry)",
                }.get(pore_label, "Irregular pore shape")
                c1, c2 = st.columns(2)
                c1.metric("Label", pore_label)
                c2.metric("Walls", f"{n_peaks}")
                st.caption(shape_desc)
                if pore_result.get("fig_theta_proj"):
                    st.caption("θ-axis Projection")
                    st.pyplot(pore_result["fig_theta_proj"])

            else:
                st.warning(pore_result["message"])

    else:
        st.info("No matching entries found.")


with tab_infomation:
    st.header("Information")
    st.write(f"理想的场景是，支持从对称性/手性角度为用户分析PDB/CIF,为用户推荐匹配的Host-Guest pair. 当用户提供手性分子时, 从数据库中为用户匹配或筛除尺寸/形态符合的CIF集合.")
    st.write("ToDo:" \
    "\n - PDB返回质心、增加compare molecules" \
    "\n - 补充CoReCOF datasheet-扩增对称信息" \
    "\n - 增加tab-生成膜系统packmol.inp" \
    "\n - 增加tab-生成Figures for analysis" \
    "\n - 增加批量操作")
    st.write("More Features: \n - tab2: 修正cpa.py,补充质量约定" \
    "\n - tab2: 基于CPA系计算分子当前朝向" \
    "\n - tab2: 基于当前朝向计算分子姿态的范数距离")

    st.write("Email: \n - Mingrui.ZUO19@student.xjtlu.edu.cn \n - Lifeng.Ding@xjtlu.edu.cn")