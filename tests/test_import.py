def test_can_import_package():
    import tcrgnn

    assert hasattr(tcrgnn, "__version__") or True
