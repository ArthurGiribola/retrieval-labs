"""One error shape for every response: ``{"error": {"code": ..., "message": ...}}``.

Messages are written here, never taken from exceptions, so no stack trace, path, SQL or
uploaded text can reach a client.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.message = message


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump())


HTTP_CODES = {404: "not_found", 405: "method_not_allowed", 413: "file_too_large"}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(request: Request, error: ApiError) -> JSONResponse:
        return error_response(error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        # Field locations and messages only: pydantic's "input" would echo the request back.
        problems = "; ".join(
            f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in error.errors()
        )
        return error_response(422, "invalid_request", problems or "invalid request")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        code = HTTP_CODES.get(error.status_code, "http_error")
        return error_response(error.status_code, code, code.replace("_", " "))

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        logger.error("unhandled error", extra={"path": request.url.path}, exc_info=error)
        return error_response(500, "internal_error", "internal error")
