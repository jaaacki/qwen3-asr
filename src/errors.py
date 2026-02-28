from fastapi.responses import JSONResponse
from logger import get_request_id


def error_response(code: str, message: str, status_code: int, **context) -> JSONResponse:
    """Build a standardized error JSONResponse."""
    req_id = get_request_id()
    ctx = dict(context) if context else {}
    if req_id:
        ctx["requestId"] = req_id
    body = {
        "code": code,
        "message": message,
        "statusCode": status_code,
    }
    if ctx:
        body["context"] = ctx
    return JSONResponse(status_code=status_code, content=body)
