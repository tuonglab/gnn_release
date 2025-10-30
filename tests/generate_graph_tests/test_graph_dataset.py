# tests/test_multigraph_dataset.py
import io
import tarfile
from pathlib import Path

import pytest
import torch

from tcrgnn.graph_gen.graph_dataset import CANCEROUS, CONTROL, MultiGraphDataset


@pytest.fixture
def aa_map():
    # minimal believable 3-letter to 1-letter mapping
    return {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F"}


@pytest.fixture
def pca_path(tmp_path: Path):
    txt = """\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13
A\t-0.11933681368554915\t4.766417488282507\t17.116718091857734\t2.2957386696193227\t3.1534658133386597\t3.641930599677665\t-2.8574722434099216\t-0.5109847142605208\t3.842872013980179\t-1.5538908427060254\t0.7608576907313682\t1.7279181034667328\t2.143332106532767\t2.080794178176028
L\t17.827236139495916\t2.574881647528705\t11.241191250613007\t6.959180659468284\t-1.9816867883332019\t2.1809343793069695\t-0.09841773290148256\t1.0903477599704852\t0.5596563930841166\t5.462774828781248\t7.368276656424866\t-1.5950365586337094\t0.6615003173623032\t0.08345277609022358
R\t-8.18842865429069\t-15.25509054246837\t-1.5225827577184192\t0.07671053034882896\t-7.413312436459251\t-8.284450135263487\t-6.454110133943519\t-5.3050681266572255\t-0.09043574063604466\t-5.391594264803815\t5.400477216943234\t-1.503589086477907\t-2.78943438011411\t-0.03986911114346333
K\t-11.83205064343963\t-13.376451068477456\t5.349271886413479\t1.9433363433549626\t-5.254241123487879\t-2.2590147482305647\t-3.6907582384935265\t0.05831584947278775\t-1.1707513016437752\t9.122410299317735\t-4.000842993810381\t2.1009446688296545\t-0.7891875430079385\t3.080767564130371
N\t-15.229827451099421\t-0.08428994287098832\t-3.488523988539807\t-6.198217844216213\t-1.827574417521372\t1.607748098227221\t4.176736861841238\t5.4001738395458725\t-0.9649569456493416\t4.007921230383572\t1.7739369527555604\t-1.3370207204773743\t-4.847987954119177\t-3.707883761189499
M\t16.174028532540678\t-4.974743513146818\t3.1971864740304823\t-1.917001989845153\t6.43208548508261\t4.222104383621662\t-5.937734817263073\t4.277949734375214\t-0.5052338560181758\t-3.1751964614569563\t-3.8687151261228006\t-6.9150539063945935\t-2.011987086101428\t2.9020424894018433
D\t-18.2146026635901\t-3.2865542528492107\t-0.40013720876500003\t-1.9044386608764203\t4.706532391207613\t4.241386042121793\t9.203847891429364\t-2.9393393580995886\t-1.9915380025111236\t-2.311473981276562\t1.3815146386149353\t-1.9293613268319334\t-3.442823266955304\t5.299530695749413
F\t19.523668935978723\t-0.29160635722853373\t-3.9640755640635494\t1.7871123702247982\t-2.3007183740805064\t3.777835251904991\t1.3699785615271916\t2.015798836924127\t-0.5623672005566894\t-1.6188732505630177\t3.58753547362929\t-1.6859013366577702\t-0.18920712621685834\t-3.7368675588168543
C\t8.263782073274541\t8.355436763283517\t-5.918452287478518\t-14.414201802910007\t12.698390941968125\t-6.393774682844861\t-3.5241563819552266\t-4.560058137275353\t-2.210810896179773\t4.122489674532446\t2.4152894167203987\t0.5756545997162357\t1.1078165871072982\t-0.31685467491914343
P\t-16.66954545108497\t11.992156909083274\t-16.75680557219609\t19.067662631345794\t6.366499708993196\t-0.2647766423864671\t-4.297000274639497\t0.4568908745990787\t-0.8316209524144414\t0.5452173753493633\t0.3356884858406183\t0.2798999924948404\t-0.296780004509173\t0.2803354322176457
Q\t-7.922921937567095\t-9.0429386536254\t1.5461512794392451\t-0.09756198597428309\t1.8288892750470165\t-1.923284232739145\t-1.5365361864684364\t0.719812752605038\t0.2765283657682689\t0.2117922109910444\t-5.484690949197341\t-3.7238190706117287\t2.5151995395192115\t-5.164519321479565
S\t-12.769938704149274\t7.932507850069988\t1.887682401814343\t-3.161877342987959\t-1.1680521513508795\t-3.8708189196824914\t2.2535704536897168\t3.4107550575881813\t6.022973047078656\t1.2697981865505972\t1.200917950973849\t-1.8607067511502964\t1.5018402300318063\t1.2904740154241034
E\t-11.877065522025996\t-12.303881229034266\t9.952960653160748\t4.223001009935872\t7.29364062858223\t3.485780896372769\t4.368565552368977\t-3.960438918350983\t-2.6670962353979304\t-1.666579905812477\t0.2155482714576777\t2.2994300724226617\t3.5436058633224286\t-3.825374389232729
T\t-4.804668298705792\t5.85893950977598\t0.09468381641365185\t-0.5874712215261503\t-0.9433776216253805\t-7.513406233549507\t3.4663232536817126\t1.7509426976881624\t7.821876966114661\t-2.8765287453831805\t-1.875653652289995\t0.1425853995414572\t0.3809589523521641\t-0.84761433486092
G\t-16.75314739664002\t21.39941767659005\t4.065766137441669\t-6.994186751501663\t-9.43962708628567\t8.440660957574387\t-4.986331568234021\t-3.3238171229873332\t-3.0400926568172655\t-1.9943267237455304\t-1.6593775206205077\t0.460904352920942\t0.2743844828297239\t-1.212167864590481
W\t17.521869580725017\t-7.78074030357549\t-13.126272029940449\t-1.7994159502103322\t-1.1387993784397783\t8.96350046864646\t0.1673671838241367\t-5.877160694411354\t7.748346298668698\t1.4370149915982888\t-2.13465036599978\t3.609472927934859\t-1.7905737134837931\t-0.22268318975673534
H\t-0.32695246533382577\t-7.987242879773412\t-4.847742554172263\t-5.024293485236312\t1.0174758113872284\t1.4976560161011347\t-2.6755706897548537\t9.511080099063028\t-2.333218989013605\t-3.7842872044066227\t1.645518934238619\t8.014484715794426\t1.2626263900862074\t1.2321201945067763
Y\t8.380229852827613\t-2.447287013480423\t-12.783121547220567\t-1.225151148278751\t-8.051939329597648\t-0.203105321667883\t4.295410180894555\t-0.3298747031223242\t-3.3565247270316303\t0.424970857852793\t-0.37312153586795443\t-3.20110893656403\t7.9266647369637475\t2.8810445768979474
I\t20.920933166625097\t5.634658939029473\t2.3799333409122503\t4.3069808057799985\t-2.4629695004430117\t-3.1236283654782437\t3.3133198128245414\t-0.547989779886314\t-4.239542812906307\t-0.010663162195674128\t-3.6635965408919704\t1.2272942942619258\t-4.5782480594035695\t-1.0206297796900896
V\t16.09673772014479\t8.316408972886924\t5.976168177998044\t2.6640951634854124\t-1.5146818479820758\t-8.223277811712373\t3.4429685149821445\t-1.3373359467809838\t-2.3080627679184618\t-2.22097511300724\t-3.024913003529677\t3.3130085664155997\t-0.5817000721963168\t0.963902063085134
"""
    p = tmp_path / "pca_encoding.txt"
    p.write_text(txt)
    return p


def make_edge_txt(dir_path: Path, name: str, content: str = "u v\nv w\n") -> Path:
    f = dir_path / name
    f.write_text(content)
    return f


def make_tar_gz_with_txts(dir_path: Path, name: str, inner_files: list[str]) -> Path:
    tar_gz = dir_path / name
    memfile = io.BytesIO()
    with tarfile.open(fileobj=memfile, mode="w:gz") as tar:
        for fname in inner_files:
            data = f"u v\n{fname} x\n"
            info = tarfile.TarInfo(name=fname)
            encoded = data.encode("utf-8")
            info.size = len(encoded)
            tar.addfile(info, io.BytesIO(encoded))
    tar_gz.write_bytes(memfile.getvalue())
    return tar_gz


def test_raw_and_processed_file_names_normalization(
    tmp_path: Path, aa_map, pca_path, monkeypatch
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    txt_a = make_edge_txt(raw_dir, "edges_a.txt")
    txt_b = make_edge_txt(raw_dir, "edges_b.txt")
    tar = make_tar_gz_with_txts(raw_dir, "batch.tar.gz", ["g1.txt", "g2.txt"])

    # put files under root/raw because PyG Dataset expects that
    root = tmp_path
    for p in [txt_a, txt_b, tar]:
        target = root / "raw" / p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())

    # stub non essential heavy parts of process
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.build_graph_from_edgelist",
        lambda *a, **k: torch.tensor([1]),
    )
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.parse_edges",
        lambda p: [("u", "v"), ("v", "w")],
    )
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.list_edge_txts",
        lambda extracted: [extracted / "g1.txt", extracted / "g2.txt"],
    )

    def _extract(_tar_path, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "g1.txt").write_text("a b\n")
        (outdir / "g2.txt").write_text("b c\n")
        return outdir

    monkeypatch.setattr("tcrgnn.graph_gen.graph_dataset.safe_extract_tar_gz", _extract)
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.tmp_root", lambda: tmp_path / "tmp"
    )
    monkeypatch.setattr("tcrgnn.graph_gen.graph_dataset.cleanup", lambda _: None)

    ds = MultiGraphDataset(
        root=str(root),
        samples=[txt_a, txt_b, tar],
        pca_path=str(pca_path),
        aa_map=aa_map,
        cancer=False,
    )

    # raw names are basenames only
    assert ds.raw_file_names == ["edges_a.txt", "edges_b.txt", "batch.tar.gz"]

    # processed names strip .tar.gz correctly
    assert ds.processed_file_names == ["edges_a.pt", "edges_b.pt", "batch.pt"]


def test_process_single_txt_and_len_get(tmp_path: Path, aa_map, pca_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    txt = make_edge_txt(raw_dir, "one.txt", "u v\nv w\n")

    # relocate into root/raw
    root = tmp_path
    (root / "raw" / txt.name).parent.mkdir(parents=True, exist_ok=True)
    (root / "raw" / txt.name).write_text(txt.read_text())

    calls = {"labels": [], "paths": []}

    def fake_build_graph(edges, pca, aa_map_local, label):
        calls["labels"].append(label)
        calls["paths"].append(len(edges))
        return {"edges": edges, "label": label}

    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.build_graph_from_edgelist", fake_build_graph
    )
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.parse_edges", lambda p: [("u", "v"), ("v", "w")]
    )
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.tmp_root", lambda: tmp_path / "tmp"
    )
    monkeypatch.setattr("tcrgnn.graph_gen.graph_dataset.cleanup", lambda _: None)

    ds = MultiGraphDataset(
        root=str(root),
        samples=[txt],
        pca_path=str(pca_path),
        aa_map=aa_map,
        cancer=True,
    )

    assert ds.len() == 1
    objs = ds.get(0)
    assert isinstance(objs, list)
    assert objs and objs[0]["label"] == CANCEROUS
    assert calls["labels"] == [CANCEROUS]
    assert calls["paths"] == [2]


def test_process_tar_gz_multiple_graphs(tmp_path: Path, aa_map, pca_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    tar = make_tar_gz_with_txts(raw_dir, "batch.tar.gz", ["g1.txt", "g2.txt", "g3.txt"])

    # move into root/raw for PyG Dataset
    root = tmp_path
    (root / "raw" / tar.name).parent.mkdir(parents=True, exist_ok=True)
    (root / "raw" / tar.name).write_bytes(tar.read_bytes())

    def _extract(_tar, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        for name in ["g1.txt", "g2.txt", "g3.txt"]:
            (outdir / name).write_text("a b\n")
        return outdir

    monkeypatch.setattr("tcrgnn.graph_gen.graph_dataset.safe_extract_tar_gz", _extract)
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.list_edge_txts",
        lambda extracted: sorted(extracted.glob("*.txt")),
    )
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.parse_edges", lambda p: [("a", "b")]
    )
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.tmp_root", lambda: tmp_path / "tmp"
    )
    monkeypatch.setattr("tcrgnn.graph_gen.graph_dataset.cleanup", lambda _: None)

    built = []

    def fake_build_graph(edges, pca, aa_map_local, label):
        built.append({"n_edges": len(edges), "label": label})
        return {"edges": edges, "label": label}

    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.build_graph_from_edgelist", fake_build_graph
    )

    ds = MultiGraphDataset(
        root=str(root),
        samples=[tar],
        pca_path=str(pca_path),
        aa_map=aa_map,
        cancer=False,
    )

    assert ds.len() == 1
    objs = ds.get(0)
    assert isinstance(objs, list) and len(objs) == 3
    assert all(obj["label"] == CONTROL for obj in objs)
    assert [b["n_edges"] for b in built] == [1, 1, 1]
