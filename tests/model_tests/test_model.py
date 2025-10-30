# test_gatv2.py
import torch

from tcrgnn.models.gatv2 import GATv2  # update this import to your actual path


def make_toy_graph(num_nodes=5, nfeat=8):
    # Simple undirected chain: 0-1-2-3-4
    src = torch.tensor([0, 1, 2, 3, 1, 2, 3, 4], dtype=torch.long)
    dst = torch.tensor([1, 2, 3, 4, 0, 1, 2, 3], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    x = torch.randn(num_nodes, nfeat)
    batch = torch.zeros(num_nodes, dtype=torch.long)  # single graph
    return x, edge_index, batch


def make_two_graph_batch(nfeat=8):
    # Graph A: 3 nodes in a chain 0-1-2
    src_a = torch.tensor([0, 1, 1, 2])
    dst_a = torch.tensor([1, 2, 0, 1])
    ea = torch.stack([src_a, dst_a])

    # Graph B: 2 nodes connected 0-1
    src_b = torch.tensor([0, 1])
    dst_b = torch.tensor([1, 0])
    eb = torch.stack([src_b, dst_b])

    # Offset B by size of A
    offset = 3
    eb_off = eb + offset

    edge_index = torch.cat([ea, eb_off], dim=1)

    # Features
    x = torch.randn(5, nfeat)

    # Batch vector: first 3 nodes belong to graph 0, last 2 to graph 1
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    return x, edge_index, batch


def test_forward_shape_and_no_nans():
    torch.manual_seed(0)
    nfeat, nhid, nclass = 8, 16, 3
    model = GATv2(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=0.0, temperature=1.0)
    model.eval()

    x, edge_index, batch = make_toy_graph(num_nodes=5, nfeat=nfeat)
    out = model(x, edge_index, batch)

    assert out.shape == (1, nclass)
    assert torch.isfinite(out).all()


def test_batch_two_graphs_produces_two_logits():
    torch.manual_seed(1)
    nfeat, nhid, nclass = 8, 32, 4
    model = GATv2(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=0.0)
    model.eval()

    x, edge_index, batch = make_two_graph_batch(nfeat=nfeat)
    out = model(x, edge_index, batch)
    assert out.shape == (2, nclass)


def test_temperature_scaling_in_eval_mode():
    torch.manual_seed(2)
    nfeat, nhid, nclass = 8, 16, 5

    base = GATv2(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=0.0, temperature=1.0)
    scaled = GATv2(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=0.0, temperature=2.0)
    # Make weights identical
    scaled.load_state_dict(base.state_dict())

    base.eval()
    scaled.eval()

    x, edge_index, batch = make_toy_graph(num_nodes=6, nfeat=nfeat)
    out_base = base(x, edge_index, batch)
    out_scaled = scaled(x, edge_index, batch)

    # Temperature=2 halves logits
    assert torch.allclose(out_scaled, out_base / 2.0, rtol=1e-5, atol=1e-6)


def test_training_mode_uses_dropout_stochastically():
    # Very high dropout to maximize chance of different outputs
    torch.manual_seed(3)
    nfeat, nhid, nclass = 8, 16, 3
    model = GATv2(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=0.9)
    model.train()

    x, edge_index, batch = make_toy_graph(num_nodes=8, nfeat=nfeat)

    torch.manual_seed(4)
    out1 = model(x, edge_index, batch)

    torch.manual_seed(5)
    out2 = model(x, edge_index, batch)

    # With different seeds and dropout on, outputs should differ
    assert not torch.allclose(out1, out2, rtol=1e-5, atol=1e-6)


def test_backward_and_gradients_exist():
    torch.manual_seed(6)
    nfeat, nhid, nclass = 10, 24, 7
    model = GATv2(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=0.1)
    model.train()

    x, edge_index, batch = make_two_graph_batch(nfeat=nfeat)  # batch size 2
    logits = model(x, edge_index, batch)

    # Make up labels
    y = torch.tensor([1, 3], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits, y)

    loss.backward()

    # Check that at least one parameter got gradients, and they are finite
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_config_concat_false_implies_classifier_in_features_eq_nhid():
    nfeat, nhid, nclass = 8, 12, 2
    model = GATv2(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=0.0)
    # Sanity: classifier should match nhid because concat=False in both GATv2Conv layers
    assert model.classifier.in_features == nhid
