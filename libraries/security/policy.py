"""The front door's policy decision: the pod-local OPA evaluates foundry/frontdoor/decision for every user
message (inbound), every reply (output) and every admin API call (admin). No answer is a refusal."""

import httpx

from libraries.observability.logger import get_console_logger

logger = get_console_logger("security.policy")

UNAVAILABLE = {
    "allowed": False,
    "verdict": "deny",
    "controls": [],
    "reasons": ["policy_decision_unavailable"],
    "obligations": [],
}


class FrontDoor:
    def __init__(self, url: str):
        self.url = url

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    async def decide(
        self,
        point: str,
        user_token: str,
        connection_id: str = "",
        message_id: str = "",
        traceparent: str = "",
        text: str = "",
        disclosures_shown: list[str] | None = None,
    ) -> dict:
        payload = {
            "input": {
                "point": point,
                "user_token": user_token,
                "connection_id": connection_id,
                "message_id": message_id,
                "traceparent": traceparent,
                "text": text,
                "disclosures_shown": disclosures_shown or [],
            }
        }
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.post(self.url, json=payload)
            response.raise_for_status()
            return response.json()["result"]
        except (httpx.HTTPError, ValueError, KeyError) as e:
            logger.error(f"Front door decision unavailable for {point}: {e!r}")
            return UNAVAILABLE


def notices(decision: dict) -> list[str]:
    return [o["notice"] for o in decision["obligations"] if o["type"] == "disclose"]
