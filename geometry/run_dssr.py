#!/usr/bin/env python3
"""
run_dssr.py
===========

Regenerates the DSSR pair-frame files from the structures under data/.

The repository ships the pair-frame files that the reported geometry was built
from, so this script is not needed to reproduce any number. It exists for two
reasons: the one external step can be repeated without typing the command by
hand on three different shells, and a fresh run can be checked against the
shipped files byte for byte.

DSSR is a separate program under its own licence and is not distributed here.
Get it from http://x3dna.org/, then put the executable on PATH, set DSSR_EXE
in the environment, or pass --dssr. The reported frames came from DSSR
v2.9.2-2026jul29; a different version may write different frames, and the
comparison printed at the end is what tells you.

USAGE
-----
  python run_dssr.py                       # all nine structures
  python run_dssr.py --dssr /opt/x3dna/bin/x3dna-dssr
  python run_dssr.py --only 6c63 6bp
  python run_dssr.py --out PATH
  python run_dssr.py --args "analyze -i={input} --more"

Output goes to out/pairframes. data/ is never written to.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
BUILDS = DATA / "construct_builds"
DEFAULT_OUT = ROOT / "out" / "pairframes"

CONSTRUCTS = (6, 8, 10, 12, 14, 16, 18)

# key, structure, name of the pair-frame file as the geometry build expects it
JOBS = [("6c63", DATA / "6C63_A.pdb", "6c63_dssr-pairFrames.txt"),
        ("5ns3", DATA / "5NS3.pdb", "5ns3-pairFrames.txt")]
JOBS += [(f"{n}bp", BUILDS / f"{n}bp.pdb", f"{n}bp-pairFrames.txt")
         for n in CONSTRUCTS]

CANDIDATES = ("x3dna-dssr", "dssr", "x3dna-dssr.exe", "dssr.exe")
DEFAULT_ARGS = "analyze -i={input} --more"


def find_dssr(explicit):
    """Locate the DSSR executable on any of the three platforms."""
    for name in [explicit, os.environ.get("DSSR_EXE")]:
        if not name:
            continue
        found = shutil.which(name)
        if found:
            return found
        if Path(name).is_file():
            return str(Path(name).resolve())
        raise SystemExit(f"DSSR not found at: {name}")
    for name in CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit(
        "DSSR is not on PATH. Pass --dssr /path/to/x3dna-dssr, or set\n"
        "DSSR_EXE in the environment. DSSR is not distributed with this\n"
        "repository; see http://x3dna.org/ .")


def run_one(exe, argv_template, structure, out_name, out_dir):
    """Run DSSR on one structure in a scratch directory and keep one file."""
    with tempfile.TemporaryDirectory() as tmp:
        argv = [exe]
        for token in argv_template.split():
            argv.append(token.replace("{input}", str(structure)))
        proc = subprocess.run(argv, cwd=tmp, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"   DSSR exited {proc.returncode}")
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
            for line in tail:
                print("   " + line)
            return None
        hits = sorted(Path(tmp).glob("*pairFrames*"))
        if len(hits) != 1:
            print(f"   expected one pairFrames file, found {len(hits)}")
            for h in hits:
                print("   " + h.name)
            return None
        dest = out_dir / out_name
        shutil.copyfile(hits[0], dest)
        return dest


def main():
    ap = argparse.ArgumentParser(
        description="Regenerate the DSSR pair-frame files from data/.")
    ap.add_argument("--dssr", default=None,
                    help="path to the DSSR executable, or its name on PATH")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output directory (default: out/pairframes)")
    ap.add_argument("--only", nargs="+", default=None, metavar="KEY",
                    help="run a subset: 6c63, 5ns3, 6bp ... 18bp")
    ap.add_argument("--args", default=DEFAULT_ARGS,
                    help=f"argument template (default: {DEFAULT_ARGS!r}); "
                         "{input} is replaced by the structure path")
    args = ap.parse_args()

    exe = find_dssr(args.dssr)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ver = subprocess.run([exe, "--version"], capture_output=True, text=True)
    version = (ver.stdout or ver.stderr or "").strip().splitlines()
    print("DSSR     ", exe)
    print("version  ", version[0] if version else "not reported")
    print("output   ", out_dir)
    print("reported geometry used v2.9.2-2026jul29\n")

    jobs = JOBS if not args.only else [j for j in JOBS if j[0] in set(args.only)]
    if not jobs:
        raise SystemExit("--only matched nothing; keys are: "
                         + ", ".join(j[0] for j in JOBS))

    same, differ, failed, absent = [], [], [], []
    for key, structure, out_name in jobs:
        if not structure.is_file():
            print(f"{key:6s} structure missing: {structure}")
            failed.append(key)
            continue
        print(f"{key:6s} {structure.name}")
        dest = run_one(exe, args.args, structure, out_name, out_dir)
        if dest is None:
            failed.append(key)
            continue
        shipped = DATA / out_name
        if not shipped.is_file():
            absent.append(key)
            print(f"   wrote {out_name}, no shipped copy to compare")
            continue
        if dest.read_bytes() == shipped.read_bytes():
            same.append(key)
            print(f"   wrote {out_name}, identical to data/{out_name}")
        else:
            differ.append(key)
            print(f"   wrote {out_name}, DIFFERS from data/{out_name}")

    print("\nidentical to the shipped files :", ", ".join(same) or "none")
    if differ:
        print("different from the shipped files:", ", ".join(differ))
        print("The shipped files are what the reported geometry was built")
        print("from. Do not copy a differing file into data/ without deciding")
        print("which one the analysis should rest on.")
    if absent:
        print("no shipped copy                :", ", ".join(absent))
    if failed:
        print("failed                         :", ", ".join(failed))
    return 1 if (differ or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
