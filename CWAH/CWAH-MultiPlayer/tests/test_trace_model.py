"""Standard-library tests for the TRACE data model (no simulator required)."""

import json
import unittest

from trace_tf import (
    DEFAULT_SCHEMA,
    BeliefSchema,
    BeliefState,
    DimensionBelief,
    DimensionSpec,
    Evidence,
    EvidenceSource,
)


class TraceModelTests(unittest.TestCase):
    def setUp(self):
        self.message = Evidence(
            source=EvidenceSource.TEAMMATE_MESSAGE,
            step=3,
            summary="Bob reported seeing a cup in the kitchen",
            reference="message:3:agent2",
        )

    def test_default_dimensions_start_unknown(self):
        belief = BeliefState(teammate_id=2)
        self.assertEqual(len(DEFAULT_SCHEMA.names()), 6)
        self.assertEqual(belief.revision, 0)
        self.assertTrue(all(value.status == "unknown" for value in belief.dimensions.values()))
        self.assertTrue(all(value.confidence is None for value in belief.dimensions.values()))
        self.assertTrue(all(value.uncertainty is None for value in belief.dimensions.values()))

    def test_update_preserves_prior_snapshot_and_evidence_history(self):
        initial = BeliefState(teammate_id=2)
        claim = {"object": "cup", "room": "kitchen"}
        first = initial.with_evidence("information", claim, 0.7, [self.message], uncertainty=0.3)
        claim["room"] = "bedroom"
        self.assertEqual(first.dimensions["information"].value["room"], "kitchen")
        self.assertEqual(initial.dimensions["information"].status, "unknown")

        visible = Evidence(EvidenceSource.VISIBLE_OBSERVATION, 4, "Cup visible in kitchen")
        second = first.with_evidence("information", "cup in kitchen", 0.95, [visible])
        self.assertEqual(second.revision, 2)
        self.assertEqual(len(second.dimensions["information"].evidence), 2)
        self.assertEqual(first.dimensions["information"].value["room"], "kitchen")
        self.assertIsNone(second.dimensions["information"].uncertainty)

    def test_unknown_can_be_restored_with_new_evidence(self):
        first = BeliefState(2).with_evidence("intent", "search kitchen", 0.6, [self.message])
        contradiction = Evidence(EvidenceSource.VISIBLE_OBSERVATION, 5, "Teammate left the kitchen")
        second = first.with_unknown("intent", [contradiction])
        self.assertEqual(second.dimensions["intent"].status, "unknown")
        self.assertIsNone(second.dimensions["intent"].value)
        self.assertEqual(len(second.dimensions["intent"].evidence), 2)

    def test_json_round_trip_and_custom_schema(self):
        schema = BeliefSchema.define("test-v1", [DimensionSpec("task_relevance", "Current relevance")])
        initial = BeliefState(teammate_id=2, schema=schema)
        updated = initial.with_evidence("task_relevance", ["cup", "plate"], 0.8, [self.message])
        restored = BeliefState.from_json(updated.to_json())
        self.assertEqual(restored.to_dict(), updated.to_dict())
        self.assertEqual(restored.schema.names(), ("task_relevance",))
        self.assertEqual(json.loads(updated.to_json())["revision"], 1)

    def test_reject_unsupported_or_unattributed_estimate(self):
        with self.assertRaises(ValueError):
            BeliefState(2).with_evidence("preference", "collaborative", 0.9, [])
        with self.assertRaises(ValueError):
            DimensionBelief(status="supported", value="collaborative", confidence=0.9)
        with self.assertRaises(ValueError):
            BeliefState(2).with_evidence("preference", "collaborative", 1.1, [self.message])
        with self.assertRaises(TypeError):
            Evidence("hidden_profile", 1, "not a legal source")
        with self.assertRaises(ValueError):
            BeliefSchema.define("duplicate", [DimensionSpec("x", "one"), DimensionSpec("x", "two")])
        with self.assertRaises(ValueError):
            BeliefState(2).with_evidence("information", {"not_json": object()}, 0.7, [self.message])

    def test_json_output_does_not_expose_mutable_storage(self):
        state = BeliefState(2).with_evidence("information", {"room": "kitchen"}, 0.7, [self.message])
        encoded = state.to_dict()
        encoded["dimensions"]["information"]["value"]["room"] = "bedroom"
        self.assertEqual(state.dimensions["information"].value["room"], "kitchen")
        exposed = state.dimensions["information"].value
        exposed["room"] = "bedroom"
        self.assertEqual(state.dimensions["information"].value["room"], "kitchen")
        with self.assertRaises(TypeError):
            state.dimensions["information"] = DimensionBelief()


if __name__ == "__main__":
    unittest.main()
