from dataclasses import dataclass


@dataclass
class RuleResult:
    incident_detected: bool
    incident_types: list[str]
    reasons: list[str]


class SafetyRules:
    def __init__(self):
        self.enabled_rules = {
            "phone_usage": True,
            "ppe_non_compliance": True,
            "unsafe_machine_interaction": True,
            "unsafe_position": True,
        }

    def _get_observation(
        self,
        observations,
        key: str,
        default=False,
    ):
        if isinstance(observations, dict):
            return observations.get(
                key,
                default,
            )

        return getattr(
            observations,
            key,
            default,
        )

    def evaluate(
        self,
        observations,
    ) -> RuleResult:
        incidents = []
        reasons = []

        phone_usage = self._get_observation(
            observations,
            "phone_usage",
        )

        ppe_violation = self._get_observation(
            observations,
            "ppe_violation",
        )

        machine_interaction = self._get_observation(
            observations,
            "machine_interaction",
        )

        unsafe_position = self._get_observation(
            observations,
            "unsafe_position",
        )

        if (
            self.enabled_rules["phone_usage"]
            and phone_usage
        ):
            incidents.append(
                "phone_usage"
            )

            reasons.append(
                "Worker appears to be using "
                "a mobile phone while working."
            )

        if (
            self.enabled_rules[
                "unsafe_machine_interaction"
            ]
            and machine_interaction
        ):
            incidents.append(
                "unsafe_machine_interaction"
            )

            reasons.append(
                "Worker appears to be "
                "interacting with industrial "
                "machinery in an unsafe manner."
            )

        if (
            self.enabled_rules[
                "ppe_non_compliance"
            ]
            and ppe_violation
        ):
            incidents.append(
                "ppe_non_compliance"
            )

            reasons.append(
                "Required PPE appears to be "
                "missing based on visible evidence."
            )

        if (
            self.enabled_rules[
                "unsafe_position"
            ]
            and unsafe_position
        ):
            incidents.append(
                "unsafe_position"
            )

            reasons.append(
                "Worker appears to be in "
                "an unsafe position."
            )

        return RuleResult(
            incident_detected=(
                len(incidents) > 0
            ),
            incident_types=incidents,
            reasons=reasons,
        )