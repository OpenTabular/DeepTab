"""Tests for search space."""


def test_get_search_space_importable():
    """get_search_space is importable from deeptab.hpo."""
    from deeptab.hpo import get_search_space


def test_hpo_all_contains_get_search_space():
    """deeptab.hpo.__all__ contains get_search_space."""
    import deeptab.hpo as hpo

    assert "get_search_space" in hpo.__all__


def test_get_search_space_returns_pair():
    """get_search_space returns a 2-tuple (param_names, param_space)."""
    from deeptab.configs import MLPConfig
    from deeptab.hpo import get_search_space

    result = get_search_space(MLPConfig())
    assert isinstance(result, tuple) and len(result) == 2


def test_get_search_space_nonempty():
    """get_search_space returns non-empty lists for a standard config."""
    from deeptab.configs import MLPConfig
    from deeptab.hpo import get_search_space

    names, space = get_search_space(MLPConfig())
    assert len(names) > 0
    assert len(space) > 0


def test_get_search_space_parallel_lengths():
    """param_names and param_space must have the same length."""
    from deeptab.configs import MLPConfig
    from deeptab.hpo import get_search_space

    names, space = get_search_space(MLPConfig())
    assert len(names) == len(space)


def test_get_search_space_names_are_strings():
    """Every element in param_names is a string."""
    from deeptab.configs import MLPConfig
    from deeptab.hpo import get_search_space

    names, _ = get_search_space(MLPConfig())
    assert all(isinstance(n, str) for n in names)


def test_get_search_space_fixed_params_excluded():
    """Parameters listed in fixed_params do not appear in the returned names."""
    from deeptab.configs import MLPConfig
    from deeptab.hpo import get_search_space

    fixed = {"dropout": 0.1}
    names, _ = get_search_space(MLPConfig(), fixed_params=fixed)
    assert "dropout" not in names


def test_get_search_space_custom_overrides():
    """A custom_search_space entry replaces the default for that parameter."""
    from skopt.space import Real

    from deeptab.configs import MLPConfig
    from deeptab.hpo import get_search_space

    custom = {"lr": Real(1e-5, 1e-3, prior="log-uniform")}
    names, space = get_search_space(MLPConfig(), custom_search_space=custom)
    if "lr" in names:
        idx = names.index("lr")
        assert isinstance(space[idx], Real)


def test_boolean_norm_is_searched_over_on_and_off():
    """ResNet's ``norm`` is a bool: searching layer names would always mean True."""
    from skopt.space import Categorical

    from deeptab.configs import ResNetConfig
    from deeptab.hpo import get_search_space

    names, space = get_search_space(ResNetConfig())
    dimension = space[names.index("norm")]
    assert isinstance(dimension, Categorical)
    assert set(dimension.categories) == {True, False}


def test_named_norm_is_still_searched_over_layer_names():
    from deeptab.configs import MambularConfig
    from deeptab.hpo import get_search_space

    names, space = get_search_space(MambularConfig())
    assert set(space[names.index("norm")].categories) == {"LayerNorm", "RMSNorm"}


def test_custom_norm_dimension_overrides_the_boolean_default():
    from skopt.space import Categorical

    from deeptab.configs import ResNetConfig
    from deeptab.hpo import get_search_space

    names, space = get_search_space(ResNetConfig(), custom_search_space={"norm": Categorical([True])})
    assert space[names.index("norm")].categories == (True,)
