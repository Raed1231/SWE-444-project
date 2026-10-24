"""Structural routing regressions; these tests never contact an AI provider."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import kb_routing as routing


class KnowledgeBaseTests(unittest.TestCase):
    EDUCATION = "svc.education.complaint"
    INTEGRITY = "svc.investment.integrity"

    def test_existing_routes_have_owned_https_channels(self):
        for service_id in routing.SERVICES:
            with self.subTest(service_id=service_id):
                service, authority, channel = routing.validate_service(service_id)
                self.assertEqual(service["authority_id"], authority["id"])
                self.assertEqual(channel["authority_id"], authority["id"])
                self.assertTrue(channel["url"].startswith("https://"))
                self.assertEqual(channel["verification_state"], "REPORTED_OBSERVED_NOT_RECHECKED")

    def test_unknown_and_nonstring_service_ids_are_rejected(self):
        for identifier in ("svc.fabricated", "", None, [], 1):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                routing.validate_service(identifier)

    def test_coverage_guidance_and_readiness_fail_closed(self):
        for key in ("coverage_status", "guidance_suitability", "routing_readiness"):
            service = deepcopy(routing.SERVICES[self.EDUCATION])
            if key == "routing_readiness":
                service[key]["status"] = "UNREVIEWED_NEW_STATE"
            else:
                service[key] = "UNREVIEWED_NEW_STATE"
            with self.subTest(key=key), patch.dict(routing.SERVICES, {self.EDUCATION: service}):
                with self.assertRaises(ValueError):
                    routing.validate_service(self.EDUCATION)
                projected = routing._prompt_knowledge()
                self.assertNotIn(self.EDUCATION, [item["id"] for item in projected["services"]])

    def test_missing_authority_channel_or_scope_is_rejected(self):
        for key, value in (("authority_id", "auth.missing"), ("entry_channel_id", "ch.missing"), ("scope_description", " ")):
            service = deepcopy(routing.SERVICES[self.EDUCATION])
            service[key] = value
            with self.subTest(key=key), patch.dict(routing.SERVICES, {self.EDUCATION: service}), self.assertRaises(ValueError):
                routing.validate_service(self.EDUCATION)

    def test_wrong_channel_authority_is_rejected(self):
        channel_id = routing.SERVICES[self.EDUCATION]["entry_channel_id"]
        channel = deepcopy(routing.CHANNELS[channel_id])
        channel["authority_id"] = "auth.moh"
        with patch.dict(routing.CHANNELS, {channel_id: channel}), self.assertRaises(ValueError):
            routing.validate_service(self.EDUCATION)

    def test_missing_insecure_or_injected_urls_are_rejected(self):
        channel_id = routing.SERVICES[self.EDUCATION]["entry_channel_id"]
        for url in (None, "", "http://example.test/", "javascript:alert(1)", "https://user:pass@example.test/", "https:///empty", "https://example.test:wrong/", "https://example.test/\nheader"):
            channel = deepcopy(routing.CHANNELS[channel_id])
            channel["url"] = url
            with self.subTest(url=url), patch.dict(routing.CHANNELS, {channel_id: channel}), self.assertRaises(ValueError):
                routing.validate_service(self.EDUCATION)

    def test_nonofficial_or_undocumented_channels_are_rejected(self):
        channel_id = routing.SERVICES[self.EDUCATION]["entry_channel_id"]
        for key in ("official_status", "usage_status"):
            channel = deepcopy(routing.CHANNELS[channel_id])
            channel[key] = "UNKNOWN"
            with self.subTest(key=key), patch.dict(routing.CHANNELS, {channel_id: channel}), self.assertRaises(ValueError):
                routing.validate_service(self.EDUCATION)

    def test_broken_or_unrelated_evidence_is_rejected(self):
        cases = [
            ("supported_scenario_ids", ["scn.fabricated"]),
            ("supported_scenario_ids", ["scn.media.gmedia_01"]),
            ("routing_rule_ids", ["rr.fabricated"]),
            ("routing_rule_ids", ["rr.media.001"]),
        ]
        for key, identifiers in cases:
            service = deepcopy(routing.SERVICES[self.EDUCATION])
            service[key] = identifiers
            with self.subTest(key=key, identifiers=identifiers), patch.dict(routing.SERVICES, {self.EDUCATION: service}), self.assertRaises(ValueError):
                routing.validate_service(self.EDUCATION)

    def test_unknown_official_eligibility_does_not_ban_documented_routing(self):
        service, _, _ = routing.validate_service(self.EDUCATION)
        self.assertIsNone(service["eligibility"]["criteria"])
        self.assertEqual(service["eligibility"]["knowledge_state"], "UNKNOWN")
        self.assertFalse(service["routing_readiness"]["production_routing_validated"])

    def test_scenarioless_investment_routes_keep_distinction_evidence(self):
        for service_id in ("svc.investment.service_contact", self.INTEGRITY):
            service, _, _ = routing.validate_service(service_id)
            self.assertEqual(service["supported_scenario_ids"], [])
            self.assertIn("rr.investment.route_distinction", routing.evidence_ids_for(service_id))
        rule = next(item for item in routing.PROMPT_KNOWLEDGE["routing_rules"] if item["id"] == "rr.investment.route_distinction")
        self.assertEqual(set(rule["service_ids"]), {"svc.investment.service_contact", self.INTEGRITY})
        self.assertIsNone(rule["conditions"]["exact_integrity_type_list"])

    def test_education_evidence_retains_actual_source_ids(self):
        self.assertEqual(routing.evidence_ids_for(self.EDUCATION), ["scn.education.moe_001", "rr.education.001"])

    def test_projection_keeps_policy_limits_and_evidence_caveats(self):
        projection = routing.PROMPT_KNOWLEDGE
        self.assertEqual(projection["decision_policy"], routing.KB["decision_policy"])
        self.assertEqual(projection["verification"], routing.KB["verification"])
        self.assertEqual(len(projection["services"]), len(routing.SERVICES))
        media = next(item for item in projection["complaint_scenarios"] if item["id"] == "scn.media.gmedia_08")
        self.assertEqual(media["routing_use_assessment"]["status"], "CLARIFY_DOMAIN_IF_NOT_EXPLICIT")
        health = next(item for item in projection["complaint_scenarios"] if item["id"] == "scn.health.moh_937_tech_001")
        self.assertIn("OTHER_OPERATOR_PATHS_NOT_TESTED", health["taxonomy_alignment_basis"])
        for service in projection["services"]:
            original = routing.SERVICES[service["id"]]
            self.assertEqual(service["scope_description"], original["scope_description"])
            self.assertEqual(service["routing_readiness"], original["routing_readiness"])
            self.assertEqual(service["documented_exclusions"], original["documented_exclusions"])

    def test_projection_omits_metadata_urls_and_keyword_arrays(self):
        serialized = json.dumps(routing.PROMPT_KNOWLEDGE, ensure_ascii=False)
        self.assertNotIn('"provenance"', serialized)
        self.assertNotIn('"keywords_ar"', serialized)
        self.assertNotIn('"keywords_en"', serialized)
        self.assertNotIn("https://", serialized)
        current = {key: routing.KB[key] for key in ("services", "routing_rules", "complaint_scenarios", "decision_policy")}
        self.assertLess(len(serialized), len(json.dumps(current, ensure_ascii=False)))

    def test_source_records_are_not_modified_by_projection(self):
        before = deepcopy(routing.KB)
        routing._prompt_knowledge()
        self.assertEqual(routing.KB, before)


if __name__ == "__main__":
    unittest.main()
