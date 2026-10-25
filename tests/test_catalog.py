from astropy.table import Table


def test_catalog_from_table(small_image):
    from visualens import Catalog

    table = Table(
        {
            "ra": [47.233, 47.234],
            "dec": [26.760, 26.761],
            "A_IMAGE": [2.0, 2.5],
            "B_IMAGE": [1.0, 1.2],
            "THETA_IMAGE": [0.0, 45.0],
        }
    )
    catalog = Catalog(table, image=small_image, verbose=False)

    assert len(catalog.cat) == 2
    assert "x" in catalog.cat.colnames
    assert "y" in catalog.cat.colnames
    assert catalog.image is small_image
