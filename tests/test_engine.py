"""
Unit tests for the SIF Precursor Engine.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from engine.preprocessor  import expand_abbreviations, normalize_text, score_signals, extract_entities
from engine.sif_classifier import classify_report, _keyword_sif_score
from engine.lsr_tagger     import tag_report
from data.synthetic_reports import generate_dataset


# ── Preprocessor Tests ─────────────────────────────────────────────────────────
class TestPreprocessor:

    def test_abbreviation_expansion(self):
        text = "Worker did not follow LOTO before PTW was issued."
        result = expand_abbreviations(text)
        assert "lockout tagout" in result.lower()
        assert "permit to work" in result.lower()

    def test_h2s_expansion(self):
        result = expand_abbreviations("H2S levels were high.")
        assert "hydrogen sulphide" in result.lower()

    def test_normalize_text_lowercase(self):
        result = normalize_text("WORKER ENTERED Tank WITHOUT Permit")
        assert result == result.lower()

    def test_normalize_whitespace(self):
        result = normalize_text("Worker   entered    tank")
        assert "  " not in result

    def test_signal_scores_high_energy(self):
        text = "Electrical shock from live panel. Pressure release from pressurized line."
        scores = score_signals(text)
        assert scores["energy_hazard"] > 0

    def test_signal_scores_barrier_failure(self):
        text = "No lockout was applied. Worker bypassed the interlock."
        scores = score_signals(text)
        assert scores["barrier_failure"] > 0

    def test_signal_scores_near_fatal(self):
        text = "Could have been fatal. This was a near miss near the wellhead."
        scores = score_signals(text)
        assert scores["near_fatal"] > 0

    def test_entity_extraction_equipment(self):
        text = "The pump and compressor were not isolated."
        entities = extract_entities(text)
        assert "pump" in entities["equipment"] or "compressor" in entities["equipment"]

    def test_entity_extraction_activity(self):
        text = "Worker was welding near the pipeline."
        entities = extract_entities(text)
        assert "welding" in entities["activity"]


# ── SIF Classifier Tests ────────────────────────────────────────────────────────
class TestSIFClassifier:

    def test_obvious_sif_report(self):
        text = (
            "worker entered confined tank without atmospheric test. "
            "h2s concentration was 45 ppm. no standby man. no permit to work."
        )
        result = classify_report(text)
        assert result["sif_potential"] is True or result["sif_score"] >= 0.1
        assert result["sif_score"] >= 0.1
        assert result["confidence"] in ("HIGH", "MEDIUM", "LOW")

    def test_obvious_non_sif_report(self):
        text = "hard hat tilted at angle. worker was reminded and corrected the issue."
        result = classify_report(text)
        # Low energy signals
        assert result["sif_score"] < 0.6

    def test_sif_result_has_required_keys(self):
        result = classify_report("some text about safety")
        assert "sif_score" in result
        assert "sif_potential" in result
        assert "confidence" in result
        assert "top_signals" in result
        assert "explanation" in result

    def test_sif_score_range(self):
        for text in [
            "electrical shock from live busbar, no lockout applied",
            "floor was slightly wet near the entrance",
            "crane overloaded, sling failed, load dropped",
        ]:
            result = classify_report(text)
            assert 0.0 <= result["sif_score"] <= 1.0

    def test_energy_isolation_sif(self):
        text = (
            "technician started working on live electrical panel without lockout tagout. "
            "energy isolation was not verified. severe electric shock occurred."
        )
        result = classify_report(text)
        kw_score, signals = _keyword_sif_score(text.lower())
        assert kw_score > 0.2

    def test_confidence_levels(self):
        high_text  = "explosion occurred due to h2s ignition in confined space. fatal potential."
        low_text   = "cable drum slightly misplaced in storage area."
        high_result = classify_report(high_text)
        low_result  = classify_report(low_text)
        assert high_result["sif_score"] >= low_result["sif_score"]


# ── LSR Tagger Tests ────────────────────────────────────────────────────────────
class TestLSRTagger:

    def test_confined_space_detection(self):
        text = "worker entered confined space tank without atmospheric test. h2s detected."
        result = tag_report(text)
        assert result["primary_rule"] == "Confined Space"
        assert "tagged_rules" in result
        assert len(result["tagged_rules"]) >= 1

    def test_hot_work_detection(self):
        text = "welding commenced near flammable vapours without hot work permit."
        result = tag_report(text)
        assert result["primary_rule"] == "Hot Work"

    def test_height_detection(self):
        text = "worker fell from scaffold at 8 metres height. no harness was worn."
        result = tag_report(text)
        assert result["primary_rule"] == "Working at Height"

    def test_energy_isolation_detection(self):
        text = "loto was not applied before maintenance. electrician received electric shock."
        result = tag_report(text)
        assert result["primary_rule"] == "Energy Isolation"

    def test_lifting_detection(self):
        text = "crane sling failed during lifting operation. load dropped on worker."
        result = tag_report(text)
        assert result["primary_rule"] == "Safe Mechanical Lifting"

    def test_driving_detection(self):
        text = "vehicle was speeding on access track. driver not wearing seatbelt. collision occurred."
        result = tag_report(text)
        assert result["primary_rule"] == "Driving"

    def test_result_structure(self):
        result = tag_report("some safety observation text")
        assert "primary_rule" in result
        assert "primary_score" in result
        assert "tagged_rules" in result
        assert "all_rules" in result
        assert isinstance(result["all_rules"], list)


# ── Data Generator Tests ────────────────────────────────────────────────────────
class TestSyntheticData:

    def test_generate_correct_count(self):
        df = generate_dataset(100)
        assert len(df) == 100

    def test_sif_label_ratio(self):
        df = generate_dataset(200)
        sif_ratio = df["sif_label"].mean()
        # Should be approximately 40% SIF
        assert 0.25 <= sif_ratio <= 0.55

    def test_required_columns_present(self):
        df = generate_dataset(50)
        required_cols = ["report_id", "date", "site", "department", "report_type", "narrative", "sif_label"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_narrative_not_empty(self):
        df = generate_dataset(50)
        assert df["narrative"].isna().sum() == 0
        assert (df["narrative"].str.len() > 20).all()

    def test_all_sites_present(self):
        df = generate_dataset(500)
        assert df["site"].nunique() >= 4

    def test_no_duplicate_report_ids(self):
        df = generate_dataset(200)
        assert df["report_id"].nunique() == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
