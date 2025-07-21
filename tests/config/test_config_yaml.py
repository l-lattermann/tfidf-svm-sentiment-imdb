import yaml
import itertools
from src.config.paths import TRAINING_PARAMS


def test_training_params_loading_and_validation():
    with open(TRAINING_PARAMS) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    vec_grid = config["vectorizer_param_grid"]
    grid_params = config["grid_search_params"]

    # ─── Validate Types in Vectorizer Grid ────────────────────────────────────────
    assert all(isinstance(n, tuple) for n in vec_grid["ngram_range"]), "ngram_range must contain tuples"
    assert all(isinstance(mf, int) for mf in vec_grid["max_features"]), "max_features must be int"
    assert all(isinstance(x, bool) for x in vec_grid["lowercase"]), "lowercase must be bool"
    assert isinstance(vec_grid["stop_words"], list) or isinstance(vec_grid["stop_words"], str)

    # ─── Validate Types in Grid Search ────────────────────────────────────────────
    assert all(isinstance(c, (int, float)) for c in grid_params["C"]), "C must be numeric (int or float)"
    assert all(isinstance(tol, float) for tol in grid_params["tol"]), "tol must be float"
    assert all(isinstance(mi, int) for mi in grid_params["max_iter"]), "max_iter must be int"
    assert all(x is None or isinstance(x, str) for x in grid_params["class_weight"]), "class_weight must be str or None"

    # ─── Generate and Check Grid Combinations ─────────────────────────────────────
    vec_keys, vec_values = zip(*vec_grid.items())
    vec_combinations = list(itertools.product(*vec_values))
    assert len(vec_combinations) > 0

    clf_keys, clf_values = zip(*grid_params.items())
    clf_combinations = list(itertools.product(*clf_values))
    assert len(clf_combinations) > 0

    # Optional: expected sizes
    assert len(vec_combinations) == 3 * 3 * 1 * 1 * 1 * 2  # 18
    assert len(clf_combinations) == 3 * 3 * 2 * 2 * 2         # 36