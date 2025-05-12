import auralbench


def test_version_contract():
    assert auralbench.__version__ == "0.1.0"
    assert auralbench.__all__ == ["__version__"]
