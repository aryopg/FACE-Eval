"""Tests for the shared model registry the figure scripts read."""

from __future__ import annotations

import pytest

from src.utils.plotting import (
    DIR_FAMILY,
    EFFORT_VARIANTS,
    FAMILY_COLORS,
    FAMILY_ORDER,
    MODEL_FAMILY,
    MODEL_LABEL,
    MODEL_LABEL_INLINE,
    MODEL_PARAMS,
    highest_effort_variants,
    select_models,
    sort_models,
    sorted_effort_variants,
)

_DIR_NAMES = [m.replace("/", "_") for m in MODEL_FAMILY]


# ---------------------------------------------------------------------------
# Registry completeness — a half-registered model sorts to the front with
# params 0 and renders under its raw directory name.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", _DIR_NAMES)
def test_every_model_has_params_and_labels(model):
    assert model in MODEL_PARAMS
    assert model in MODEL_LABEL
    assert model in MODEL_LABEL_INLINE


def test_every_family_has_a_colour_and_a_place_in_the_order():
    families = set(MODEL_FAMILY.values())
    assert families <= set(FAMILY_COLORS)
    assert families <= set(FAMILY_ORDER)


def test_effort_variant_dirs_inherit_their_base_family():
    # A variant suffix the DIR_FAMILY loop does not cover would fall through to
    # "Other" and render grey, sorted last.
    for base, variants in EFFORT_VARIANTS.items():
        for variant in variants:
            assert DIR_FAMILY.get(variant) == DIR_FAMILY[base], variant


# ---------------------------------------------------------------------------
# select_models
# ---------------------------------------------------------------------------


def test_orders_by_family_then_params():
    models = ["openai_gpt-oss-120b", "Qwen_Qwen3.5-9B", "openai_gpt-oss-20b", "Qwen_Qwen3.5-4B"]
    assert select_models(models) == [
        "Qwen_Qwen3.5-4B",
        "Qwen_Qwen3.5-9B",
        "openai_gpt-oss-20b",
        "openai_gpt-oss-120b",
    ]


def test_drops_models_absent_from_the_registry():
    assert select_models(["Qwen_Qwen3.5-4B", "some_unreleased_model"]) == ["Qwen_Qwen3.5-4B"]


def test_restricted_family_list_leaves_a_family_out():
    # Models come and go from the registry, so pick the family to drop from it.
    dropped = FAMILY_ORDER[-1]
    models = [m for m in _DIR_NAMES if DIR_FAMILY[m] in (FAMILY_ORDER[0], dropped)]
    kept = select_models(models, [f for f in FAMILY_ORDER if f != dropped])
    assert kept == [m for m in select_models(models) if DIR_FAMILY[m] != dropped]
    assert kept != select_models(models)


def test_effort_suffixed_dirs_carry_their_family():
    assert DIR_FAMILY["openai_gpt-oss-120b_high"] == "GPT-OSS"
    assert DIR_FAMILY["thinkingmachines_Inkling-NVFP4_0.7"] == "Inkling"


def test_highest_effort_keeps_one_variant_per_model():
    models = [
        "openai_gpt-oss-20b_low",
        "openai_gpt-oss-20b_medium",
        "openai_gpt-oss-20b_high",
        "zai-org_GLM-5.2-FP8_high",
        "zai-org_GLM-5.2-FP8_max",
    ]
    assert highest_effort_variants(models) == ["openai_gpt-oss-20b_high", "zai-org_GLM-5.2-FP8_max"]


def test_highest_effort_compares_inkling_floats_numerically():
    # "0.7" > "0.99" as strings, so a lexical comparison would pick the wrong one.
    variants = ["thinkingmachines_Inkling-NVFP4_0.7", "thinkingmachines_Inkling-NVFP4_0.99"]
    assert highest_effort_variants(variants) == ["thinkingmachines_Inkling-NVFP4_0.99"]


def test_highest_effort_passes_through_models_with_no_effort_axis():
    models = ["Qwen_Qwen3.5-9B", "moonshotai_Kimi-K2.6"]
    assert highest_effort_variants(models) == sort_models(models)


def test_sorted_effort_variants_orders_lowest_first():
    assert sorted_effort_variants("openai_gpt-oss-20b") == [
        "openai_gpt-oss-20b_low",
        "openai_gpt-oss-20b_medium",
        "openai_gpt-oss-20b_high",
        "openai_gpt-oss-20b_max",
    ]
    assert sorted_effort_variants("zai-org_GLM-5.2-FP8") == ["zai-org_GLM-5.2-FP8_high", "zai-org_GLM-5.2-FP8_max"]


def test_sorted_effort_variants_matches_the_highest_effort_pick():
    for base in EFFORT_VARIANTS:
        assert sorted_effort_variants(base)[-1] == highest_effort_variants(EFFORT_VARIANTS[base])[0]
