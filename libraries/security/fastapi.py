import jwt
from fastapi import HTTPException, Request

from .keycloak import Caller, Keycloak


def require_bearer(keycloak: Keycloak):
    async def dependency(request: Request) -> Caller | None:
        if not keycloak.enabled or request.url.path.endswith("/health"):
            return None
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer token", headers={"WWW-Authenticate": "Bearer"})
        try:
            return keycloak.validate(auth[7:].strip())
        except jwt.PyJWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}", headers={"WWW-Authenticate": "Bearer"})

    return dependency
