"""Stateless, owned-model-only public deployment boundary.

Research factories and adapter configuration remain in mapexploc.api. This
factory deliberately never consults either research model environment variable.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .api import AnalyzeRequest, RequestLimits, create_app
from .default_model import resolve_default_model, source_root
from .methods import MethodConfiguration

logger = logging.getLogger(__name__)
WEB_LIMITS = RequestLimits(
    prediction_proteins=25,
    prediction_residues=50_000,
    explanation_proteins=5,
    explanation_residues=10_000,
)
MAX_REQUEST_BYTES = 512_000
MAX_RESPONSE_BYTES = 4_000_000  # Leave room below Vercel's 4.5 MB payload limit.


def _analysis_policy(request: AnalyzeRequest) -> None:
    """Bound optional report work as well as the sequence batch."""
    if len(request.annotations) > 200:
        raise HTTPException(413, "Live reports accept at most 200 annotations")
    config = request.configuration
    if isinstance(config, MethodConfiguration) and (
        config.bootstrap_replicates > 1000
        or config.reference_sensitivity
        or config.faithfulness
        or config.reference_pool
    ):
        raise HTTPException(
            422,
            "Live reports support at most 1,000 bootstrap replicates and no "
            "experimental reference or intervention runs; use the research CLI",
        )


class _PublicBoundary:
    """Bound streamed bodies before parsing; log request outcomes without input."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.monotonic()
        request_id = uuid4().hex
        status = 500
        oversized_response = False
        size_error = json.dumps(
            {
                "detail": "Report exceeds the public response limit. Accept gzip "
                "compression, use fewer proteins, or generate the report with the CLI."
            }
        ).encode()

        async def observed_send(message: Message) -> None:
            nonlocal status, oversized_response
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = dict(message.get("headers", []))
                # Shared API responses are materialized JSON with Content-Length.
                # GZipMiddleware has already compressed negotiated responses.
                if int(headers.get(b"content-length", b"0")) > MAX_RESPONSE_BYTES:
                    status = 413
                    oversized_response = True
                    message = {
                        "type": "http.response.start",
                        "status": status,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(size_error)).encode()),
                        ],
                    }
                message["headers"] = [
                    *message.get("headers", []),
                    (b"x-request-id", request_id.encode("ascii")),
                    (b"cache-control", b"no-store"),
                ]
            elif message["type"] == "http.response.body" and oversized_response:
                if message.get("more_body", False):
                    return
                message = {"type": "http.response.body", "body": size_error}
            await send(message)

        try:
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                    response = JSONResponse(
                        status_code=413,
                        content={"detail": "Request body exceeds 512,000 bytes"},
                    )
                    await response(scope, receive, observed_send)
                    return
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
            delivered = False

            async def bounded_receive() -> Message:
                nonlocal delivered
                if delivered:
                    return await receive()
                delivered = True
                return {"type": "http.request", "body": bytes(body)}

            await self.app(scope, bounded_receive, observed_send)
        finally:
            logger.info(
                json.dumps(
                    {
                        "event": "web_request",
                        "request_id": request_id,
                        "method": scope["method"],
                        "status": status,
                        "duration_ms": round((time.monotonic() - started) * 1000),
                    }
                )
            )


def create_web_app(*, trusted_root: Path | None = None) -> FastAPI:
    """Serve the release-approved manifest at /api, ignoring operator adapters.

    Vercel's entrypoint passes its own bundle root explicitly so an installed
    wheel does not need a source-checkout layout or a particular cwd.
    """
    root = trusted_root if trusted_root is not None else source_root()
    if root is None:
        raise ValueError("The public application requires a trusted model bundle")
    service = create_app(
        model_selection=resolve_default_model(trusted_root=root),
        limits=WEB_LIMITS,
        analysis_policy=_analysis_policy,
    )

    @service.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail
        if exc.status_code >= 500:
            logger.error(
                "Public API operation failed",
                exc_info=exc,
                extra={"event": "web_operation_failed", "status": exc.status_code},
            )
            detail = "The model service is temporarily unavailable. Please retry."
        # v2 prediction exceptions can contain adapter implementation details.
        elif exc.status_code == 422:
            detail = "Request does not satisfy the supported public model contract"
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})

    @service.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
                    for error in exc.errors()
                ]
            },
        )

    @service.exception_handler(Exception)
    async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.error("Public API failed", exc_info=exc, extra={"event": "web_error"})
        return JSONResponse(
            status_code=500, content={"detail": "The analysis could not be completed"}
        )

    application = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    application.mount("/api", service)
    application.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
    application.add_middleware(_PublicBoundary)
    return application
