def test_image_empty_placeholder(small_image):
    img = small_image

    assert img.image_path is None
    assert img.image_data.shape == (64, 64)
    assert img.wcs is not None
    assert img.pix_deg_scale > 0
