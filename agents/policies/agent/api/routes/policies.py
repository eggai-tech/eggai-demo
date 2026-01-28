from fastapi import APIRouter, HTTPException, Query

from agents.policies.agent.api.models import (
    PersonalPolicy,
    PolicyListResponse,
)
from libraries.observability.logger import get_console_logger

logger = get_console_logger("policies_api_routes")

router = APIRouter()


@router.get("/policies", response_model=PolicyListResponse)
async def list_personal_policies(
    category: str | None = Query(None, description="Filter by category (auto, home, life)"),
    limit: int = Query(20, description="Number of policies to return", ge=1, le=100),
    offset: int = Query(0, description="Offset for pagination", ge=0),
):
    try:
        from agents.policies.agent.tools.database.policy_data import get_all_policies

        all_policies = get_all_policies()

        if category:
            filtered_policies = [p for p in all_policies if p["policy_category"] == category]
        else:
            filtered_policies = all_policies

        paginated_policies = filtered_policies[offset:offset + limit]
        policies = [PersonalPolicy(**policy) for policy in paginated_policies]

        return PolicyListResponse(
            policies=policies,
            total=len(filtered_policies)
        )
    except Exception as e:
        logger.error(f"List policies error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error while listing policies"
        )


@router.get("/policies/{policy_number}", response_model=PersonalPolicy)
async def get_personal_policy(policy_number: str):
    try:
        from agents.policies.agent.tools.database.policy_data import (
            get_personal_policy_details,
        )

        policy_json = get_personal_policy_details(policy_number)

        if policy_json == "Policy not found.":
            raise HTTPException(
                status_code=404, detail=f"Policy not found: {policy_number}"
            )

        import json
        policy_data = json.loads(policy_json)

        return PersonalPolicy(**policy_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get policy error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal server error while retrieving policy"
        )
