
from agents.triage.classifiers.registry import (
    compare_classifiers,
    get_available_versions,
    list_classifiers,
)


class TestRegistry:

    def test_list_classifiers(self):
        classifiers = list_classifiers()
        assert len(classifiers) == 9  # v0 through v8

        versions = {c.version for c in classifiers}
        assert versions == {"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"}

    def test_get_available_versions(self):
        versions = get_available_versions()
        assert versions == ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"]

    def test_compare_classifiers_all(self):
        comparison = compare_classifiers()
        assert len(comparison) == 9
        assert "v0" in comparison
        assert "v4" in comparison
        assert "v8" in comparison

    def test_compare_classifiers_subset(self):
        comparison = compare_classifiers(["v0", "v4", "v6"])
        assert len(comparison) == 3
        assert "v0" in comparison
        assert "v4" in comparison
        assert "v6" in comparison
        assert "v1" not in comparison

    def test_classifier_info_fields(self):
        classifiers = list_classifiers()

        for info in classifiers:
            assert info.version is not None
            assert info.name is not None
            assert info.description is not None
            assert isinstance(info.requires_llm, bool)
            assert isinstance(info.requires_training, bool)
            assert isinstance(info.trainable, bool)
            assert len(info.estimated_latency_ms) == 2

    def test_v4_is_default_recommendation(self):
        classifiers = list_classifiers()
        v4 = next(c for c in classifiers if c.version == "v4")
        assert "Recommended" in v4.description or "default" in v4.description.lower()

    def test_llm_requirements(self):
        comparison = compare_classifiers()

        # These require LLM
        assert comparison["v0"].requires_llm is True
        assert comparison["v4"].requires_llm is True
        assert comparison["v6"].requires_llm is True

        # These don't require LLM (local models)
        assert comparison["v3"].requires_llm is False
        assert comparison["v5"].requires_llm is False
        assert comparison["v7"].requires_llm is False
        assert comparison["v8"].requires_llm is False
