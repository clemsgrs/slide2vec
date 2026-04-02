import importlib.util


def test_core_import_surface_is_direct():
    import slide2vec
    import slide2vec.api
    import slide2vec.encoders
    import slide2vec.inference
    from slide2vec.runtime_types import LoadedModel

    assert hasattr(slide2vec, "Model")
    assert slide2vec.api.LoadedModel is LoadedModel
    assert slide2vec.inference.LoadedModel is LoadedModel
    assert importlib.util.find_spec("slide2vec.encoders.compat") is None
