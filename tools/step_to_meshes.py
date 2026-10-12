"""Tessellate every named solid of a STEP file into binary STL files + a JSON manifest (name, volume, centroid, bbox, mm).

Runs in the cq-editor environment (OCP / cadquery):  <cq python> tools/step_to_meshes.py <file.step> <out_dir> [deflection_mm]
Names come from the XCAF document (product / component names), transforms from the assembly structure.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib


def label_name(label):
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return attr.Get().ToExtString()
    return ""


def tessellate(shape, deflection):
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.3, True)
    V, F = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            base = len(V)
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(trsf)
                V.append((p.X(), p.Y(), p.Z()))
            rev = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                F.append((base + a - 1, base + c - 1, base + b - 1) if rev else (base + a - 1, base + b - 1, base + c - 1))
        exp.Next()
    return np.array(V, float), np.array(F, int)


def write_stl(path, V, F):
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(F)))
        for a, b, c in F:
            p, q, r = V[a], V[b], V[c]
            n = np.cross(q - p, r - p)
            nn = np.linalg.norm(n)
            n = n / nn if nn > 0 else n
            f.write(struct.pack("<12fH", *n, *p, *q, *r, 0))


def main(step, out_dir, deflection=0.15):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(step)) != IFSelect_RetDone:
        raise SystemExit(f"cannot read {step}")
    reader.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    free = TDF_LabelSequence()
    st.GetFreeShapes(free)
    items = []

    def visit(label, loc, path):
        name = label_name(label)
        if st.IsAssembly_s(label):
            comps = TDF_LabelSequence()
            st.GetComponents_s(label, comps)
            for i in range(1, comps.Length() + 1):
                visit(comps.Value(i), loc, path + [name])
            return
        if st.IsReference_s(label):
            ref = TDF_Label()
            st.GetReferredShape_s(label, ref)
            loc2 = loc.Multiplied(st.GetLocation_s(label))
            visit(ref, loc2, path + [name] if name else path)
            return
        shape = st.GetShape_s(label)
        shape = shape.Moved(loc)
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        k = 0
        while exp.More():
            solid = exp.Current()
            items.append((name or "unnamed", path, solid, k))
            k += 1
            exp.Next()
        if k == 0 and not st.IsAssembly_s(label):
            items.append((name or "unnamed", path, shape, 0))

    for i in range(1, free.Length() + 1):
        visit(free.Value(i), TopLoc_Location(), [])

    manifest = []
    used = {}
    for name, path, solid, k in items:
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(solid, props)
        vol = props.Mass()
        c = props.CentreOfMass()
        V, F = tessellate(solid, deflection)
        if len(V) == 0:
            print(f"skipping {name}: no geometry")
            continue
        xmin, ymin, zmin = V.min(0)
        xmax, ymax, zmax = V.max(0)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:60]
        n = used.get(safe, 0)
        used[safe] = n + 1
        fname = f"{safe}" + (f"_{n}" if n else "") + ".stl"
        write_stl(out_dir / fname, V, F)
        manifest.append(dict(name=name, path=path, file=fname, volume_mm3=vol, centroid_mm=[c.X(), c.Y(), c.Z()],
                             bbox_mm=[[xmin, ymin, zmin], [xmax, ymax, zmax]], n_vertices=int(len(V)), n_faces=int(len(F))))
        print(f"{fname:50s} vol {vol:12.1f} mm3  centroid {c.X():8.2f} {c.Y():8.2f} {c.Z():8.2f}  bbox x[{xmin:.1f},{xmax:.1f}] y[{ymin:.1f},{ymax:.1f}] z[{zmin:.1f},{zmax:.1f}]  tris {len(F)}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} solids -> {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.15)
