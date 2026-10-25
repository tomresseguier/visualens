def test_import_visualens():
    import visualens

    assert visualens.Image is not None
    assert visualens.Catalog is not None
    assert visualens.LensModel is not None
    assert "Image" in visualens.__all__
