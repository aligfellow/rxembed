import rxembed


def test_import():
    assert rxembed is not None


def test_public_surface():
    for name in ("embed", "Ensemble", "EnsembleSet", "metal", "nci_modes", "geometry", "set_verbose"):
        assert hasattr(rxembed, name)


def test_version_is_a_string():
    assert isinstance(rxembed.__version__, str)
    assert rxembed is not None
