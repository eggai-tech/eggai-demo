import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
import httpx
import jwt
import uvicorn
from eggai import eggai_cleanup
from eggai.transport import eggai_set_default_transport
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import HTMLResponse, Response

from libraries.communication.transport import create_kafka_transport
from libraries.observability.logger import get_console_logger
from libraries.observability.tracing import init_telemetry

from .config import keycloak, settings

eggai_set_default_transport(
    lambda: create_kafka_transport(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        ssl_cert=settings.kafka_ca_content,
    )
)

from .agent import add_websocket_gateway, frontend_agent

logger = get_console_logger("frontend_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await frontend_agent.start()
        logger.info(f"{settings.app_name} started successfully")

        yield
    finally:
        logger.info("Cleaning up resources")
        await eggai_cleanup()


api = FastAPI(lifespan=lifespan)


async def _serve_html(filename: str) -> HTMLResponse:
    html_file_path = Path(settings.default_public_dir) / filename
    try:
        if not html_file_path.is_file():
            raise FileNotFoundError(f"File not found: {html_file_path}")

        async with aiofiles.open(html_file_path, encoding="utf-8") as file:
            file_content = await file.read()

        return HTMLResponse(content=file_content, status_code=200)
    except FileNotFoundError as fnf_error:
        raise HTTPException(status_code=404, detail=str(fnf_error))
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@api.get("/", response_class=HTMLResponse)
async def read_root():
    return await _serve_html("index.html")


@api.get("/admin.html", response_class=HTMLResponse)
async def read_admin():
    return await _serve_html("admin.html")


@api.get("/config")
async def read_config():
    links = [
        {"name": name.strip(), "url": url.strip()}
        for name, url in (item.split("=", 1) for item in settings.platform_links.split(","))
    ]
    keycloak_config = None
    if settings.keycloak_url:
        keycloak_config = {
            "url": settings.keycloak_public_url or settings.keycloak_url,
            "realm": settings.keycloak_realm,
            "clientId": settings.keycloak_web_client_id,
        }
    return {"platformLinks": links, "keycloak": keycloak_config}


upstream = httpx.AsyncClient(timeout=30)

HOP_BY_HOP = {"host", "content-length", "content-encoding", "transfer-encoding", "connection"}


@api.api_route("/api/{agent}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_api(agent: str, path: str, request: Request):
    targets = {
        "policies": settings.api_policies_url,
        "claims": settings.api_claims_url,
        "billing": settings.api_billing_url,
    }
    if agent not in targets:
        raise HTTPException(status_code=404)
    url = f"{targets[agent]}/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    if keycloak.enabled:
        token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        try:
            caller = keycloak.validate(token)
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
        if "insurance-admin" not in caller.roles:
            raise HTTPException(status_code=403, detail="insurance-admin role required")
        headers["authorization"] = f"Bearer {await keycloak.exchange(token, f'insurance-{agent}')}"
    response = await upstream.request(request.method, url, content=await request.body(), headers=headers)
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers={k: v for k, v in response.headers.items() if k.lower() not in HOP_BY_HOP},
    )


frontend_server = uvicorn.Server(
    uvicorn.Config(
        api, host=settings.host, port=settings.port, log_level=settings.log_level
    )
)

add_websocket_gateway(settings.websocket_path, api, frontend_server)

if __name__ == "__main__":
    try:
        logger.info(f"Starting {settings.app_name}")
        init_telemetry(app_name=settings.app_name, endpoint=settings.otel_endpoint)
        logger.info(f"Telemetry initialized for {settings.app_name}")

        logger.info(f"Server starting at http://{settings.host}:{settings.port}")
        asyncio.run(frontend_server.serve())
    except KeyboardInterrupt:
        logger.info("Shutting down frontend agent")
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
