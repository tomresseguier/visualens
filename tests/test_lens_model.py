from pathlib import Path

MINIMAL_LENS_MODEL_PAR = Path(__file__).resolve().parent / "fixtures" / "minimal_model" / "minimal.par"


def test_lens_model_minimal_parameter_file(small_image):
    from visualens import LensModel

    assert MINIMAL_LENS_MODEL_PAR.is_file()

    model = LensModel(
        str(MINIMAL_LENS_MODEL_PAR.resolve()),
        image=small_image,
        compute_predictions=False,
        verbose=False,
    )

    assert model.z_lens == 0.356
    assert model.reference == (47.233278, 26.7604953)
    assert model.image is small_image
