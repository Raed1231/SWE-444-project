"""Read-only KB projection and structural checks for documented routing.

These checks establish that a route exists and is usable in this limited pilot.
They do not establish semantic complaint fit or official applicant eligibility.
"""

import json
from pathlib import Path
from urllib.parse import urlsplit


KB = json.loads(Path(__file__).with_name("wajhah_routing_kb.json").read_text(encoding="utf-8"))


def _index(records):
    indexed = {}
    for record in records:
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in indexed:
            raise ValueError("Invalid or duplicate knowledge-base identifier")
        indexed[identifier] = record
    return indexed


SERVICES = _index(KB["services"])
AUTHORITIES = _index(KB["authorities"])
CHANNELS = _index(KB["channels"])
_SCENARIOS = _index(KB["complaint_scenarios"])
_RULES = _index(KB["routing_rules"])
_TAXONOMY = _index(KB["taxonomy_values"])
KNOWN_EVIDENCE_IDS = frozenset(_SCENARIOS) | frozenset(_RULES)

# This KB authorizes conditional initial experiments, not production validation.
_COVERAGE = {"DOCUMENTED_LIMITED_SCOPE"}
_GUIDANCE = {"DOCUMENTED_WITH_LIMITATIONS"}
_READINESS = {"CONDITIONAL_INITIAL_EXPERIMENTS"}

# Fixed diagnostic vocabulary only. Never include model identifiers, complaint
# contents, channel values, or raw parser exceptions in a public/loggable error.
ROUTE_VALIDATION_REASONS = frozenset({
    'KB_UNKNOWN_SERVICE', 'KB_SERVICE_COVERAGE', 'KB_SERVICE_GUIDANCE',
    'KB_SERVICE_READINESS', 'KB_SERVICE_SCOPE', 'KB_AUTHORITY_OR_CHANNEL_MISSING',
    'KB_CHANNEL_AUTHORITY', 'KB_CHANNEL_STATUS', 'KB_CHANNEL_URL',
    'KB_CHANNEL_URL_PARSE', 'KB_CHANNEL_URL_POLICY', 'KB_CHANNEL_PORT',
    'KB_SCENARIO_REFERENCE', 'KB_RULE_REFERENCE', 'KB_ROUTE_INVALID',
})


class RouteValidationError(ValueError):
    """A safe reason code, compatible with existing ValueError consumers."""
    def __init__(self, reason_code):
        self.reason_code = reason_code if isinstance(reason_code, str) and reason_code in ROUTE_VALIDATION_REASONS else 'KB_ROUTE_INVALID'
        super().__init__(self.reason_code)


def _rule_service_ids(rule):
    identifiers = [rule.get("service_id")]
    identifiers.extend(rule.get("service_ids", []))
    for key, value in (rule.get("conditions") or {}).items():
        if key.endswith("_service_id"):
            identifiers.append(value)
    return list(dict.fromkeys(value for value in identifiers if isinstance(value, str)))


def validate_service(service_id):
    """Return (service, authority, channel), or reject an invalid pilot route."""
    if not isinstance(service_id, str) or service_id not in SERVICES:
        raise RouteValidationError('KB_UNKNOWN_SERVICE')
    service = SERVICES[service_id]
    if service.get("coverage_status") not in _COVERAGE:
        raise RouteValidationError('KB_SERVICE_COVERAGE')
    if service.get("guidance_suitability") not in _GUIDANCE:
        raise RouteValidationError('KB_SERVICE_GUIDANCE')
    if (service.get("routing_readiness") or {}).get("status") not in _READINESS:
        raise RouteValidationError('KB_SERVICE_READINESS')
    if not isinstance(service.get("scope_description"), str) or not service["scope_description"].strip():
        raise RouteValidationError('KB_SERVICE_SCOPE')
    authority = AUTHORITIES.get(service.get("authority_id"))
    channel = CHANNELS.get(service.get("entry_channel_id"))
    if authority is None or channel is None:
        raise RouteValidationError('KB_AUTHORITY_OR_CHANNEL_MISSING')
    if channel.get("authority_id") != authority["id"]:
        raise RouteValidationError('KB_CHANNEL_AUTHORITY')
    if channel.get("official_status") != "REPORTED_OFFICIAL" or channel.get("usage_status") != "DOCUMENTED":
        raise RouteValidationError('KB_CHANNEL_STATUS')
    url = channel.get("url")
    if not isinstance(url, str) or any(character.isspace() for character in url):
        raise RouteValidationError('KB_CHANNEL_URL')
    try:
        parsed = urlsplit(url)
    except ValueError:
        raise RouteValidationError('KB_CHANNEL_URL_PARSE') from None
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise RouteValidationError('KB_CHANNEL_URL_POLICY')
    # Parsing the port also rejects invalid ports rather than returning a bad link.
    try:
        parsed.port
    except ValueError:
        raise RouteValidationError('KB_CHANNEL_PORT') from None
    for scenario_id in service.get("supported_scenario_ids", []):
        scenario = _SCENARIOS.get(scenario_id)
        if scenario is None or scenario.get("service_id") != service_id or scenario.get("authority_id") != authority["id"]:
            raise RouteValidationError('KB_SCENARIO_REFERENCE')
    for rule_id in service.get("routing_rule_ids", []) + service.get("related_route_distinction_rule_ids", []):
        rule = _RULES.get(rule_id)
        if rule is None or service_id not in _rule_service_ids(rule) or rule.get("authority_id") != authority["id"]:
            raise RouteValidationError('KB_RULE_REFERENCE')
    return service, authority, channel


def evidence_ids_for(service_id):
    """List source scenarios and linked rules; no keyword-based matching occurs."""
    service, _, _ = validate_service(service_id)
    identifiers = list(service.get("supported_scenario_ids", []))
    identifiers.extend(service.get("routing_rule_ids", []))
    identifiers.extend(service.get("related_route_distinction_rule_ids", []))
    identifiers.extend(rule["id"] for rule in _RULES.values() if service_id in _rule_service_ids(rule))
    return list(dict.fromkeys(identifiers))


def _without_provenance(value):
    if isinstance(value, dict):
        return {key: _without_provenance(item) for key, item in value.items() if key != "provenance"}
    if isinstance(value, list):
        return [_without_provenance(item) for item in value]
    return value


def _prompt_knowledge():
    # Projection is built from the unchanged source on startup, never a second KB.
    projection = {
        "verification": KB["verification"],
        "decision_policy": KB["decision_policy"],
        "authorities": [
            {key: authority[key] for key in ("id", "name_ar", "name_en", "ministry_wide_coverage")}
            for authority in AUTHORITIES.values()
        ],
        "services": [],
        "routing_rules": [],
        "complaint_scenarios": [],
    }
    for service in SERVICES.values():
        try:
            validate_service(service["id"])
        except ValueError:
            continue
        projected = {
            key: _without_provenance(value)
            for key, value in service.items()
            if key not in {"provenance", "entry_channel_id", "platform_channel_id", "observed_scope_option_ids"}
        }
        if service.get("observed_scope_option_ids"):
            projected["observed_scope_options"] = [
                _without_provenance(_TAXONOMY[identifier]) for identifier in service["observed_scope_option_ids"]
            ]
        projection["services"].append(projected)
    eligible_ids = {service["id"] for service in projection["services"]}
    for rule in _RULES.values():
        service_ids = [identifier for identifier in _rule_service_ids(rule) if identifier in eligible_ids]
        if not service_ids:
            continue
        projected = {
            key: _without_provenance(value)
            for key, value in rule.items()
            if key not in {"provenance", "channel_id", "keywords_ar", "keywords_en", "keywords_are_hints_not_sufficient_conditions"}
        }
        projected["service_ids"] = service_ids
        projection["routing_rules"].append(projected)
    projection["complaint_scenarios"] = [
        {
            key: _without_provenance(value)
            for key, value in scenario.items()
            if key not in {"provenance", "channel_id", "taxonomy_value_ids", "original_scenario_id"}
        }
        for scenario in _SCENARIOS.values()
        if scenario["service_id"] in eligible_ids
    ]
    return projection


PROMPT_KNOWLEDGE = _prompt_knowledge()
