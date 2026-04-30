from autocrat.settings import COT_MODE_TABLE, DATASET_REGISTRY, INFO_MODE_TABLE, validate_mode_tables


def test_mode_levels_are_complete() -> None:
    validate_mode_tables()
    assert set(INFO_MODE_TABLE) == {1, 2, 3, 4}
    assert set(COT_MODE_TABLE) == {0, 1, 2, 3, 4}
    assert [INFO_MODE_TABLE[idx].temperature for idx in (1, 2, 3, 4)] == [0.0, 0.3, 0.7, 1.0]
    assert [COT_MODE_TABLE[idx].max_thinking_tokens for idx in (0, 1, 2, 3, 4)] == [0, 64, 256, 1024, 4096]


def test_representative_dataset_categories_present() -> None:
    required = {
        "gsm8k": "math",
        "math_500": "math",
        "arc_challenge": "logic",
        "gpqa_diamond": "qa",
        "humaneval": "code",
        "mbpp": "code",
    }
    for key, category in required.items():
        assert key in DATASET_REGISTRY
        assert DATASET_REGISTRY[key].category == category
