import math
import os
from collections import OrderedDict


def write_edges(pdb_file, output_dir):
    # residue_idx → { resname: str, coord: (x,y,z) }
    residues = OrderedDict()

    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            cols = line.split()
            atom_name = cols[2]
            resname = cols[3]
            residx = int(cols[5])  # your “1…12” index
            x, y, z = map(float, cols[6:9])

            # pick CB if present, otherwise CA for glycine
            if atom_name == "CB" or (resname == "GLY" and atom_name == "CA"):
                residues.setdefault(residx, {})["coord"] = (x, y, z)
                residues[residx]["resname"] = resname

    # sort by residue index
    items = list(residues.items())  # [(1, {...}), (2, {...}), …]

    out_path = os.path.join(
        output_dir, os.path.basename(pdb_file).replace(".pdb", "_edge.txt")
    )
    with open(out_path, "w") as out:
        for i, (idx1, r1) in enumerate(items):
            for idx2, r2 in items[i + 1 :]:
                d = math.dist(r1["coord"], r2["coord"])
                if d <= 8.0:
                    out.write(f"{r1['resname']} {idx1} {r2['resname']} {idx2}\n")
