#!/usr/bin/env python3
"""
build_ruler_geometry.py
=======================

Builds the donor-acceptor geometry of the MII&Cy5 helical ruler from DSSR
base-pair reference frames, and computes the orientation factor kappa^2 and
the predicted FRET efficiency for each construct.

Base-pair normals come directly from the DSSR reference frames. Taking the
local helical direction as the vector between consecutive base-pair centres
instead would place it roughly 28 degrees away from the true normal.

WORKFLOW
--------
  1. Read the TO1-B donor (ligand EKJ) from the Mango II crystal structure.
     Donor centre  = midpoint of the two ring nitrogens N09 and N15.
     Donor dipole  = N15 -> N09, the long axis of the chromophore.

  2. Read the Cy5 acceptor (ligand 5CY) from the 5NS3 crystal structure and
     express its pose (offset + dipole) in the reference frame of the terminal
     base pair it stacks on. This makes the pose rotation invariant, so it can
     be transferred to any other terminal base pair.
     Acceptor dipole = N1 -> N2, along the pentamethine bridge.

  3. For each construct, superimpose the conserved CGUAC (the last five base
     pairs of the build) onto base pairs 2-6 of the Mango II stem, using a
     Kabsch fit on the origin and the three axis tips of every pair, i.e. the
     full six degrees of freedom of each frame, with no atom pruning.

  4. Graft the Cy5 pose onto the terminal base pair of each aligned construct.

  5. Compute the donor-acceptor distance, kappa^2, and the FRET efficiency.

INPUT FILES
-----------
  6C63_A.pdb                Mango II chain A, including ligand EKJ (TO1-B)
  6c63_dssr-pairFrames.txt  DSSR pair frames for 6C63 chain A
  {N}bp-pairFrames.txt      DSSR pair frames for each construct, N = 6..18
  5NS3.pdb                  Cy5-labelled RNA duplex, including ligand 5CY (Cy5)
  5ns3-pairFrames.txt       DSSR pair frames for 5NS3

  All are produced by:   dssr analyze -i=<file>.pdb --more

OUTPUT FILES
------------
  ruler_geometry.csv        per-construct geometry, kappa^2 and efficiency
  ruler_all_data.py         geometry in the format read by model/fret_model.py
  ruler_helix_frames.py     base-pair frame ladder of the longest construct

USAGE
-----
  python build_ruler_geometry.py [--dir PATH] [--out PATH] [--r0 55.0]

With no arguments it reads the repository's data folder and writes to the
repository's out folder, both resolved from this file's own location, so the
working directory does not matter and the same command works on Linux,
Windows and macOS.

Requires numpy.
"""

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
# Default paths are resolved from this file, never from the working directory,
# so the script behaves the same on Linux, Windows and macOS and from any
# directory. Nothing outside the repository is read or written.

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DEFAULT_IN_DIR = os.path.join(_ROOT, "data")
DEFAULT_OUT_DIR = os.path.join(_ROOT, "out")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONSTRUCTS = [6, 8, 10, 12, 14, 16, 18]

# Experimental efficiencies and replicate standard deviations
EXP_E = {6: 0.732, 8: 0.814, 10: 0.716, 12: 0.542, 14: 0.452, 16: 0.585, 18: 0.484}
EXP_SD = {6: 0.013, 8: 0.006, 10: 0.007, 12: 0.014, 14: 0.035, 16: 0.023, 18: 0.014}

# Donor: TO1-B, ligand EKJ in the Mango II structure
DONOR_LIGAND = "EKJ"
DONOR_ATOM_1 = "N15"      # quinolinium nitrogen  (dipole tail)
DONOR_ATOM_2 = "N09"      # benzothiazole nitrogen (dipole head)

# Acceptor: Cy5, ligand 5CY in the 5NS3 structure
ACCEPTOR_LIGAND = "5CY"
ACCEPTOR_CHAIN = "C"      # chain D is only partially resolved, do not use it
ACCEPTOR_ATOM_1 = "N1"    # indolium nitrogen (dipole tail)
ACCEPTOR_ATOM_2 = "N2"    # indolium nitrogen (dipole head)

# Base pair of 5NS3 that the Cy5 stacks on: DSSR pair 1, C.C26-C.G58
ACCEPTOR_REF_PAIR = 1

# Base pairs of the Mango II stem used for alignment.
# DSSR pair 1 is the open end where the extension attaches; pair 6 is adjacent
# to the loop and the G-quadruplex. The conserved CGUAC is pairs 2-6.
MANGO_ALIGN_PAIRS = [2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_pair_frames(path):
    """Read a DSSR pairFrames file.

    Returns {pair_number: dict(name, residues, o, x, y, z)} where o is the
    base-pair origin and x, y, z are the frame axes. The z axis is the
    base-pair normal.
    """
    with open(path) as handle:
        lines = handle.read().strip().splitlines()

    frames = {}
    i = 1                                   # line 0 is the "# N pair frames" header
    while i + 4 < len(lines) + 1 and i < len(lines):
        header = lines[i].split()
        number = int(header[0])
        name = header[1]
        residues = " ".join(header[2:])
        vec = lambda k: np.array([float(v) for v in lines[i + k].split()[:3]])
        frames[number] = dict(name=name, residues=residues,
                              o=vec(1), x=vec(2), y=vec(3), z=vec(4))
        i += 5
    return frames


def parse_ligand_atoms(path, resname, chain=None):
    """Read the atoms of one ligand from a PDB file.

    Returns {atom_name: xyz}. If chain is given, only that chain is read.
    """
    atoms = {}
    with open(path) as handle:
        for line in handle:
            if line[:6].strip() not in ("ATOM", "HETATM"):
                continue
            if line[17:20].strip() != resname:
                continue
            if chain is not None and line[21] != chain:
                continue
            name = line[12:16].strip()
            atoms[name] = np.array([float(line[30:38]),
                                    float(line[38:46]),
                                    float(line[46:54])])
    return atoms


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def frame_matrix(frame):
    """Matrix whose columns are the frame axes, mapping local to global."""
    return np.column_stack([frame["x"], frame["y"], frame["z"]])


def unit(v):
    return v / np.linalg.norm(v)


def kabsch(P, Q):
    """Rigid transform (R, t) that best maps point set P onto Q."""
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - R @ cp


def frame_points(frame):
    """Origin plus the three axis tips, so a Kabsch fit captures orientation."""
    o = frame["o"]
    return [o, o + frame["x"], o + frame["y"], o + frame["z"]]


def kappa_squared(mu_d, mu_a, r_vec):
    """Orientation factor from the two dipoles and their separation vector."""
    r_hat = unit(r_vec)
    return float((np.dot(mu_d, mu_a)
                  - 3.0 * np.dot(mu_d, r_hat) * np.dot(mu_a, r_hat)) ** 2)


def fret_efficiency(r, k2, r0_iso):
    """Efficiency with the Forster radius scaled by the orientation factor."""
    r0 = r0_iso * (max(k2, 1e-10) / (2.0 / 3.0)) ** (1.0 / 6.0)
    return float(1.0 / (1.0 + (r / r0) ** 6))


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def get_donor(pdb_path):
    """Donor centre and unit transition dipole, in the structure's own frame."""
    atoms = parse_ligand_atoms(pdb_path, DONOR_LIGAND)
    if not atoms:
        raise SystemExit(f"No {DONOR_LIGAND} ligand found in {pdb_path}")
    for name in (DONOR_ATOM_1, DONOR_ATOM_2):
        if name not in atoms:
            raise SystemExit(f"Donor atom {name} missing from {pdb_path}")
    tail, head = atoms[DONOR_ATOM_1], atoms[DONOR_ATOM_2]
    return (tail + head) / 2.0, unit(head - tail), np.linalg.norm(head - tail)


def get_acceptor_local_pose(pdb_path, frames_path):
    """Cy5 pose expressed in the frame of the base pair it stacks on.

    Returned as (local_offset, local_dipole). Both are invariant to the
    orientation of the parent structure, so they can be grafted onto the
    terminal base pair of any construct.
    """
    atoms = parse_ligand_atoms(pdb_path, ACCEPTOR_LIGAND, chain=ACCEPTOR_CHAIN)
    if not atoms:
        raise SystemExit(f"No {ACCEPTOR_LIGAND} ligand found in chain "
                         f"{ACCEPTOR_CHAIN} of {pdb_path}")
    for name in (ACCEPTOR_ATOM_1, ACCEPTOR_ATOM_2):
        if name not in atoms:
            raise SystemExit(f"Acceptor atom {name} missing; chain "
                             f"{ACCEPTOR_CHAIN} may be incompletely resolved")

    tail, head = atoms[ACCEPTOR_ATOM_1], atoms[ACCEPTOR_ATOM_2]
    centre = (tail + head) / 2.0
    dipole = unit(head - tail)

    frames = parse_pair_frames(frames_path)
    ref = frames[ACCEPTOR_REF_PAIR]
    M = frame_matrix(ref)

    local_offset = M.T @ (centre - ref["o"])
    local_dipole = M.T @ dipole
    return local_offset, local_dipole, np.linalg.norm(head - tail), ref


def align_build_to_mango(build_frames, mango_frames, n_bp):
    """Transform mapping one construct into the Mango II coordinate frame.

    The conserved CGUAC is the last five base pairs of the build. These are
    superimposed on Mango pairs 2-6. Each pair contributes four points, the
    origin and the three axis tips, so the fit uses the full orientation of
    every frame rather than position alone.
    """
    build_ids = list(range(n_bp - 4, n_bp + 1))
    P, Q = [], []
    for b, m in zip(build_ids, MANGO_ALIGN_PAIRS):
        P.extend(frame_points(build_frames[b]))
        Q.extend(frame_points(mango_frames[m]))
    P, Q = np.array(P), np.array(Q)

    R, t = kabsch(P, Q)
    rmsd = float(np.sqrt(np.mean(np.sum((P @ R.T + t - Q) ** 2, axis=1))))

    normal_dev = [float(np.degrees(np.arccos(np.clip(
        abs(np.dot(R @ build_frames[b]["z"], mango_frames[m]["z"])), -1, 1))))
        for b, m in zip(build_ids, MANGO_ALIGN_PAIRS)]

    return R, t, rmsd, build_ids, normal_dev


def transform_frame(frame, R, t):
    return dict(name=frame["name"], residues=frame["residues"],
                o=R @ frame["o"] + t, x=R @ frame["x"],
                y=R @ frame["y"], z=R @ frame["z"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build the MII&Cy5 ruler geometry from DSSR pair frames.")
    ap.add_argument("--dir", default=DEFAULT_IN_DIR,
                    help="directory holding the input files "
                         "(default: the repository's data folder)")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR,
                    help="directory for the output files "
                         "(default: the repository's out folder)")
    ap.add_argument("--r0", type=float, default=55.0,
                    help="isotropic Forster radius in angstrom (default: 55)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    d = args.dir
    need = ["6C63_A.pdb", "6c63_dssr-pairFrames.txt",
            "5NS3.pdb", "5ns3-pairFrames.txt"]
    need += [f"{n}bp-pairFrames.txt" for n in CONSTRUCTS]
    missing = [f for f in need if not os.path.isfile(os.path.join(d, f))]
    if missing:
        print("Missing input files in", os.path.abspath(d), file=sys.stderr)
        for f in missing:
            print("   ", f, file=sys.stderr)
        print("Pass --dir to point at the folder that holds them.",
              file=sys.stderr)
        raise SystemExit(1)

    p = lambda f: os.path.join(d, f)

    # --- donor -------------------------------------------------------------
    donor_pos, donor_dip, donor_len = get_donor(p("6C63_A.pdb"))
    print("Donor: TO1-B (ligand %s), from 6C63 chain A" % DONOR_LIGAND)
    print("   centre        %s" % np.round(donor_pos, 3))
    print("   dipole        %s   (%s -> %s)"
          % (np.round(donor_dip, 4), DONOR_ATOM_1, DONOR_ATOM_2))
    print("   N-N distance  %.2f A" % donor_len)

    # --- acceptor local pose ----------------------------------------------
    acc_off, acc_dip, acc_len, acc_ref = get_acceptor_local_pose(
        p("5NS3.pdb"), p("5ns3-pairFrames.txt"))
    print("\nAcceptor: Cy5 (ligand %s), from 5NS3 chain %s"
          % (ACCEPTOR_LIGAND, ACCEPTOR_CHAIN))
    print("   stacked on    %s  (%s)" % (acc_ref["name"], acc_ref["residues"]))
    print("   local offset  %s   |offset| = %.2f A"
          % (np.round(acc_off, 3), np.linalg.norm(acc_off)))
    print("   local dipole  %s" % np.round(acc_dip, 4))
    print("   stacking separation along the base-pair normal  %+.2f A" % acc_off[2])
    tilt = 90.0 - np.degrees(np.arccos(np.clip(abs(acc_dip[2]), -1, 1)))
    print("   dipole tilt out of the base-pair plane          %+.1f deg" % tilt)
    print("   N-N distance  %.2f A" % acc_len)

    # --- per construct -----------------------------------------------------
    mango = parse_pair_frames(p("6c63_dssr-pairFrames.txt"))
    rows, aligned_all = [], {}

    print("\nAlignment and grafting")
    print("   %-6s %-10s %-12s %-10s %-9s %-8s %-9s"
          % ("stem", "RMSD (A)", "normal (deg)", "r_DA (A)", "kappa^2",
             "E model", "E exp"))

    for n in CONSTRUCTS:
        build = parse_pair_frames(p(f"{n}bp-pairFrames.txt"))
        R, t, rmsd, build_ids, normal_dev = align_build_to_mango(build, mango, n)
        aligned = {k: transform_frame(v, R, t) for k, v in build.items()}
        aligned_all[n] = aligned

        # Pair 1 is the terminal G-C base pair, where Cy5 stacks
        term = aligned[1]
        M = frame_matrix(term)
        acc_pos = term["o"] + M @ acc_off
        acc_vec = unit(M @ acc_dip)

        r_vec = acc_pos - donor_pos
        r = float(np.linalg.norm(r_vec))
        k2 = kappa_squared(donor_dip, acc_vec, r_vec)
        e_model = fret_efficiency(r, k2, args.r0)
        e_iso = fret_efficiency(r, 2.0 / 3.0, args.r0)

        print("   %-6s %-10.3f %-12.2f %-10.2f %-9.4f %-8.4f %-9.3f"
              % (f"{n}bp", rmsd, float(np.mean(normal_dev)), r, k2,
                 e_model, EXP_E[n]))

        rows.append(dict(bp=n, rmsd=rmsd, normal_mean=float(np.mean(normal_dev)),
                         normal_max=float(np.max(normal_dev)),
                         term_o=term["o"], term_x=term["x"],
                         term_y=term["y"], term_z=term["z"],
                         acc_pos=acc_pos, acc_dip=acc_vec,
                         r=r, k2=k2, e_model=e_model, e_iso=e_iso,
                         e_exp=EXP_E[n], sd_exp=EXP_SD[n]))

    write_csv(os.path.join(args.out, "ruler_geometry.csv"),
              rows, donor_pos, donor_dip)
    write_helix_frames(os.path.join(args.out, "ruler_helix_frames.py"),
                       aligned_all, acc_off, acc_dip, donor_pos, donor_dip)
    write_all_data(os.path.join(args.out, "ruler_all_data.py"),
                   rows, donor_pos, donor_dip)
    written = ["ruler_geometry.csv", "ruler_all_data.py",
               "ruler_helix_frames.py"]
    print("\nWrote " + ", ".join(written))


def write_csv(path, rows, donor_pos, donor_dip):
    with open(path, "w", newline="\n") as f:
        f.write("# MII&Cy5 helical ruler geometry, built from DSSR pair frames\n")
        f.write("# Coordinates are in the frame of 6C63_A.pdb, in angstrom\n")
        f.write("# donor_centre,%.4f,%.4f,%.4f\n" % tuple(donor_pos))
        f.write("# donor_dipole,%.6f,%.6f,%.6f\n" % tuple(donor_dip))
        f.write("bp,align_rmsd_A,normal_dev_mean_deg,normal_dev_max_deg,"
                "term_ox,term_oy,term_oz,"
                "term_xx,term_xy,term_xz,term_yx,term_yy,term_yz,"
                "term_zx,term_zy,term_zz,"
                "cy5_x,cy5_y,cy5_z,cy5_dip_x,cy5_dip_y,cy5_dip_z,"
                "r_DA_A,kappa2,E_model,E_isotropic,E_exp,SD_exp\n")
        for r in rows:
            f.write("%d,%.4f,%.3f,%.3f," % (r["bp"], r["rmsd"],
                                            r["normal_mean"], r["normal_max"]))
            f.write("%.4f,%.4f,%.4f," % tuple(r["term_o"]))
            f.write("%.6f,%.6f,%.6f," % tuple(r["term_x"]))
            f.write("%.6f,%.6f,%.6f," % tuple(r["term_y"]))
            f.write("%.6f,%.6f,%.6f," % tuple(r["term_z"]))
            f.write("%.4f,%.4f,%.4f," % tuple(r["acc_pos"]))
            f.write("%.6f,%.6f,%.6f," % tuple(r["acc_dip"]))
            f.write("%.3f,%.5f,%.5f,%.5f,%.3f,%.3f\n"
                    % (r["r"], r["k2"], r["e_model"], r["e_iso"],
                       r["e_exp"], r["sd_exp"]))


def write_helix_frames(path, aligned_all, acc_off, acc_dip,
                       donor_pos, donor_dip):
    """Export the full base-pair frame ladder of the longest construct.

    Every construct's terminal base pair coincides with a base pair of the
    longest construct, to within about 0.3 A and 0.6 degrees, because all
    constructs describe the same continuous helix. The frame ladder below is
    therefore the geometry of that helix, sampled at integer base pairs, and
    it is what the FRET model uses to interpolate to fractional lengths.
    """
    longest = max(aligned_all)
    frames = aligned_all[longest]
    with open(path, "w", newline="\n") as f:
        f.write('"""Base-pair frame ladder of the longest construct.\n\n'
                'Generated by build_ruler_geometry.py. Coordinates are in the\n'
                'frame of 6C63_A.pdb. Construct of length N has its terminal\n'
                'base pair at ladder index (LONGEST + 1 - N).\n"""\n\n')
        f.write("LONGEST_BP = %d\n\n" % longest)
        f.write("DONOR_POS = [%.4f, %.4f, %.4f]\n" % tuple(donor_pos))
        f.write("DONOR_DIP = [%.4f, %.4f, %.4f]\n\n" % tuple(donor_dip))
        f.write("# Cy5 pose in the frame of the base pair it stacks on\n")
        f.write("CY5_LOCAL_OFFSET = [%.4f, %.4f, %.4f]\n" % tuple(acc_off))
        f.write("CY5_LOCAL_DIPOLE = [%.4f, %.4f, %.4f]\n\n" % tuple(acc_dip))
        f.write("# index: origin, x-axis, y-axis, z-axis (z is the bp normal)\n")
        f.write("HELIX_FRAMES = {\n")
        for k in sorted(frames):
            v = frames[k]
            f.write("    %d: dict(\n" % k)
            f.write("        o=[%.4f, %.4f, %.4f],\n" % tuple(v["o"]))
            f.write("        x=[%.6f, %.6f, %.6f],\n" % tuple(v["x"]))
            f.write("        y=[%.6f, %.6f, %.6f],\n" % tuple(v["y"]))
            f.write("        z=[%.6f, %.6f, %.6f]),\n" % tuple(v["z"]))
        f.write("}\n")


def write_all_data(path, rows, donor_pos, donor_dip):
    """Emit the geometry in the all_data format used by the FRET model."""
    with open(path, "w", newline="\n") as f:
        f.write('"""Geometry for the Monte Carlo FRET model.\n\n'
                'Generated by build_ruler_geometry.py from DSSR pair frames.\n'
                'Coordinates are in the frame of 6C63_A.pdb.\n"""\n\n')
        f.write("ALL_DATA = [\n")
        for r in rows:
            f.write("    {\n")
            f.write('        "bp": %d,\n' % r["bp"])
            f.write('        "R_D": [%.4f, %.4f, %.4f],\n' % tuple(donor_pos))
            f.write('        "mu_D": [%.4f, %.4f, %.4f],\n' % tuple(donor_dip))
            f.write('        "R_A": [%.4f, %.4f, %.4f],\n' % tuple(r["acc_pos"]))
            f.write('        "mu_A": [%.4f, %.4f, %.4f],\n' % tuple(r["acc_dip"]))
            f.write("    },\n")
        f.write("]\n")


if __name__ == "__main__":
    main()
