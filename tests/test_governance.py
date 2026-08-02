"""
Unit tests for XAI Governance Middleware Architecture.
Validates Level 1 (Input/Constraints), Level 2 (XAI Controls/Counterfactuals),
Level 3 (Simulatability & Illusion of Understanding), and Level 4 (Audit Trail Gate).
"""

import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.core.input_layer import load_tabular, ClinicalDataLayer
from src.methods.counterfactual import generate_counterfactual
from src.experiments.evaluation import simulatability_score, detect_illusion_of_understanding
from src.governance.audit_trail import generate_audit_record


class TestXAIGovernanceMiddleware(unittest.TestCase):

    def setUp(self):
        self.data = load_tabular(random_state=42)
        self.clf = RandomForestClassifier(n_estimators=10, random_state=42)
        self.clf.fit(self.data.X_train, self.data.y_train)

    def test_level_1_clinical_data_layer_plausibility(self):
        immutable = ["sepal length (cm)", "sepal width (cm)"]
        cdl = ClinicalDataLayer(data=self.data, immutable_features=immutable)

        inst_a = self.data.X_test[0].copy()
        inst_b_valid = inst_a.copy()
        inst_b_valid[2] += 0.5  # mutate petal length (mutable)

        inst_c_invalid = inst_a.copy()
        inst_c_invalid[0] += 0.5  # mutate sepal length (immutable)

        self.assertTrue(cdl.validate_plausibility(inst_a, inst_b_valid))
        self.assertFalse(cdl.validate_plausibility(inst_a, inst_c_invalid))

    def test_level_2_counterfactual_immutable_constraints(self):
        instance = self.data.X_test[0]
        immutable = ["sepal length (cm)"]  # index 0

        cf_res = generate_counterfactual(
            model=self.clf,
            instance=instance,
            X_train=self.data.X_train,
            feature_names=self.data.feature_names,
            immutable_features=immutable,
        )

        # Index 0 must remain exact
        self.assertTrue(np.isclose(cf_res["original"][0], cf_res["counterfactual"][0], atol=1e-4))
        self.assertIn("metrics", cf_res)
        self.assertIn("L0", cf_res["metrics"])
        self.assertIn("L1", cf_res["metrics"])
        self.assertIn("L2", cf_res["metrics"])

    def test_level_3_simulatability_and_illusion_of_understanding(self):
        # Positive simulatability
        score_pos = simulatability_score(accuracy_pre=0.50, accuracy_post=0.85)
        self.assertAlmostEqual(score_pos, 0.35, places=2)
        self.assertFalse(detect_illusion_of_understanding(score_pos))

        # Zero or negative simulatability -> Illusion of Understanding
        score_zero = simulatability_score(accuracy_pre=0.70, accuracy_post=0.70)
        self.assertEqual(score_zero, 0.0)
        self.assertTrue(detect_illusion_of_understanding(score_zero))

        score_neg = simulatability_score(accuracy_pre=0.70, accuracy_post=0.60)
        self.assertAlmostEqual(score_neg, -0.10, places=2)
        self.assertTrue(detect_illusion_of_understanding(score_neg))

    def test_level_4_compliance_audit_trail_gate(self):
        # Valid decision
        record_valid = generate_audit_record(
            model_type="RandomForest",
            model_accuracy=0.95,
            instance_id=1,
            true_label="setosa",
            predicted_label="setosa",
            xai_method="SHAP",
            feature_attributions={"petal length (cm)": 0.4},
            confidence=0.98,
            illusion_of_understanding=False,
        )

        self.assertTrue(record_valid.decision_justified)
        self.assertIn("GDPR Art. 22", record_valid.compliance_tags)
        self.assertIn("EU AI Act Art. 13", record_valid.compliance_tags)

        # Rejected decision due to illusion of understanding
        record_rejected = generate_audit_record(
            model_type="RandomForest",
            model_accuracy=0.95,
            instance_id=2,
            true_label="versicolor",
            predicted_label="versicolor",
            xai_method="LIME",
            feature_attributions={"petal length (cm)": 0.1},
            confidence=0.60,
            illusion_of_understanding=True,
        )

        self.assertFalse(record_rejected.decision_justified)
        self.assertTrue(record_rejected.illusion_of_understanding)
        self.assertIn("Audit Gate: REJECTED", record_rejected.compliance_tags)
        self.assertIn("illusion_of_understanding: True", record_rejected.compliance_tags)


if __name__ == "__main__":
    unittest.main()
