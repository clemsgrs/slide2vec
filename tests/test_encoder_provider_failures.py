"""Failed installed Encoder providers remain isolated at the public API seam."""

import subprocess
import sys
import textwrap


def _run_isolated(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


_PLUGIN_SCENARIO_PREAMBLE = """
import importlib.metadata
import warnings
import torch

class EntryPoint:
    def __init__(self, name, value, provider):
        self.name = name
        self.value = value
        self._provider = provider

    def load(self):
        return self._provider

class EntryPoints(list):
    def select(self, *, group):
        assert group == "slide2vec.encoders"
        return self

def install_providers(*providers):
    original_entry_points = importlib.metadata.entry_points
    def entry_points(**kwargs):
        if kwargs:
            return original_entry_points(**kwargs)
        return EntryPoints(providers)
    importlib.metadata.entry_points = entry_points

def test_encoder_class(encode_dim=3):
    from slide2vec.encoders import TileEncoder

    class TestEncoder(TileEncoder):
        def __init__(self, *, output_variant=None):
            self._device = torch.device("cpu")

        @property
        def encode_dim(self):
            return encode_dim

        @property
        def device(self):
            return self._device

        def to(self, device):
            self._device = torch.device(device)
            return self

        def get_transform(self):
            return lambda image: image

        def encode_tiles(self, batch):
            return batch[:, :encode_dim]

    return TestEncoder

def register_test_encoder(name, encode_dim=3):
    from slide2vec.encoders import register_encoder

    encoder_cls = test_encoder_class(encode_dim)
    register_encoder(
        name,
        output_variants={"default": {"encode_dim": encode_dim}},
        default_output_variant="default",
        input_size=224,
        supports_variable_input_size=False,
        supported_spacing_um=0.5,
        precision="fp32",
    )(encoder_cls)
    return encoder_cls
"""


def _plugin_scenario(body: str) -> str:
    return _PLUGIN_SCENARIO_PREAMBLE + textwrap.dedent(body)


def test_provider_exception_rolls_back_every_preset() -> None:
    _run_isolated(
        _plugin_scenario("""
        state = {}

        def register_encoders():
            from slide2vec import list_models
            state["before"] = list_models()
            register_test_encoder("rolled-back-preset")
            raise RuntimeError("provider setup failed")

        install_providers(
            EntryPoint(
                "raising-provider",
                "raising_plugin:register_encoders",
                register_encoders,
            )
        )

        from slide2vec import list_encoder_provider_diagnostics, list_models

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            models = list_models()

        assert models == state["before"]
        assert "rolled-back-preset" not in models
        assert [str(item.message) for item in caught] == [
            "Skipped Encoder provider 'raising-provider': RuntimeError: provider setup failed"
        ]

        diagnostics = list_encoder_provider_diagnostics()
        assert len(diagnostics) == 1
        diagnostic = diagnostics[0]
        assert diagnostic.provider_key == "raising-provider"
        assert diagnostic.provider == "raising_plugin:register_encoders"
        assert diagnostic.exception_type == "RuntimeError"
        assert diagnostic.message == "provider setup failed"
        """)
    )


def test_late_builtin_collision_rejects_the_whole_provider() -> None:
    _run_isolated(
        _plugin_scenario("""
        state = {}

        def register_encoders():
            from slide2vec.encoders import encoder_registry, register_encoder
            state["builtin_owner"] = encoder_registry.require("virchow2")
            plugin_encoder = register_test_encoder("discarded-before-collision")

            register_encoder(
                "virchow2",
                output_variants={"default": {"encode_dim": 3}},
                default_output_variant="default",
                input_size=224,
                supports_variable_input_size=False,
                supported_spacing_um=0.5,
            )(plugin_encoder)

        install_providers(
            EntryPoint("collision", "collision_plugin:register_encoders", register_encoders)
        )

        from slide2vec import list_encoder_provider_diagnostics, list_models
        from slide2vec.encoders import encoder_registry

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models = list_models()

        assert "discarded-before-collision" not in models
        assert encoder_registry.require("virchow2") is state["builtin_owner"]
        diagnostics = list_encoder_provider_diagnostics()
        assert [(item.provider_key, item.exception_type, item.message) for item in diagnostics] == [
            (
                "collision",
                "ValueError",
                "'virchow2' is already registered in the encoders registry",
            )
        ]
        """)
    )


def test_invalid_metadata_rolls_back_an_earlier_valid_preset() -> None:
    _run_isolated(
        _plugin_scenario("""
        state = {}

        def register_encoders():
            from slide2vec import list_models
            from slide2vec.encoders import register_encoder
            state["before"] = list_models()
            register_test_encoder("discarded-before-invalid-metadata")

            register_encoder(
                "invalid-metadata",
                output_variants={"embedding": {"encode_dim": 3}},
                default_output_variant="missing",
                input_size=224,
                supports_variable_input_size=False,
                supported_spacing_um=0.5,
            )

        install_providers(
            EntryPoint("invalid", "invalid_plugin:register_encoders", register_encoders)
        )

        from slide2vec import list_encoder_provider_diagnostics, list_models

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models = list_models()

        assert models == state["before"]
        assert "discarded-before-invalid-metadata" not in models
        diagnostics = list_encoder_provider_diagnostics()
        assert diagnostics[0].provider_key == "invalid"
        assert diagnostics[0].exception_type == "ValueError"
        assert diagnostics[0].message == (
            "default_output_variant 'missing' must be present in output_variants"
        )
        """)
    )


def test_healthy_and_broken_providers_coexist_through_public_lookup() -> None:
    _run_isolated(
        _plugin_scenario("""
        calls = []

        def broken_provider():
            calls.append("a-broken")
            register_test_encoder("broken-partial")
            raise ModuleNotFoundError("missing_private_dependency")

        def healthy_provider():
            calls.append("z-healthy")
            register_test_encoder("healthy-preset")

        install_providers(
            EntryPoint("z-healthy", "healthy_plugin:register_encoders", healthy_provider),
            EntryPoint("a-broken", "broken_plugin:register_encoders", broken_provider),
        )

        from dataclasses import FrozenInstanceError
        from slide2vec import Model, list_encoder_provider_diagnostics, list_models

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            models = list_models()

        assert calls == ["a-broken", "z-healthy"]
        assert "virchow2" in models
        assert "healthy-preset" in models
        assert "broken-partial" not in models
        assert [str(item.message) for item in caught] == [
            "Skipped Encoder provider 'a-broken': "
            "ModuleNotFoundError: missing_private_dependency"
        ]

        assert Model.from_preset("virchow2", device="cpu").name == "virchow2"
        healthy = Model.from_preset("healthy-preset", device="cpu")
        assert healthy.feature_dim == 3

        try:
            Model.from_preset("expected-from-broken-provider", device="cpu")
        except KeyError as error:
            missing_message = str(error)
        else:
            raise AssertionError("missing plugin preset unexpectedly resolved")
        assert "Available:" in missing_message
        assert "'a-broken' (ModuleNotFoundError: missing_private_dependency)" in missing_message

        diagnostics = list_encoder_provider_diagnostics()
        assert diagnostics == list_encoder_provider_diagnostics()
        assert isinstance(diagnostics, tuple)
        try:
            diagnostics[0].message = "changed"
        except FrozenInstanceError:
            pass
        else:
            raise AssertionError("provider diagnostic was mutable")
        """)
    )


def test_provider_diagnostics_are_sorted_by_provider_key_and_value() -> None:
    _run_isolated(
        _plugin_scenario("""
        calls = []

        def provider(label, message):
            def fail():
                calls.append(label)
                raise RuntimeError(message)
            return fail

        install_providers(
            EntryPoint("zeta", "plugins:zeta", provider("zeta", "z failed")),
            EntryPoint("alpha", "plugins:second", provider("alpha-second", "second failed")),
            EntryPoint("alpha", "plugins:first", provider("alpha-first", "first failed")),
        )

        from slide2vec import list_encoder_provider_diagnostics, list_models

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            list_models()

        assert calls == ["alpha-first", "alpha-second", "zeta"]
        diagnostics = list_encoder_provider_diagnostics()
        assert [(item.provider_key, item.provider) for item in diagnostics] == [
            ("alpha", "plugins:first"),
            ("alpha", "plugins:second"),
            ("zeta", "plugins:zeta"),
        ]
        assert [item.message for item in diagnostics] == [
            "first failed",
            "second failed",
            "z failed",
        ]
        assert [str(item.message) for item in caught] == [
            "Skipped Encoder providers 'alpha': RuntimeError: first failed; "
            "'alpha': RuntimeError: second failed; 'zeta': RuntimeError: z failed"
        ]
        """)
    )


def test_entry_point_load_failure_is_isolated_from_later_provider() -> None:
    _run_isolated(
        _plugin_scenario("""
        class UnloadableEntryPoint(EntryPoint):
            def load(self):
                register_test_encoder("import-side-effect")
                raise ModuleNotFoundError("optional_plugin_runtime")

        def later_provider():
            register_test_encoder("later-healthy", encode_dim=2)

        install_providers(
            UnloadableEntryPoint("unloadable", "unloadable_plugin:register", None),
            EntryPoint("working", "working_plugin:register", later_provider),
        )

        from slide2vec import list_encoder_provider_diagnostics, list_models

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models = list_models()

        assert "import-side-effect" not in models
        assert "later-healthy" in models
        assert [item.provider_key for item in list_encoder_provider_diagnostics()] == [
            "unloadable"
        ]
        """)
    )
