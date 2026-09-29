"""HTTP routes. Handlers stay thin: validation in, service call, response out."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ml.data.schema import Transaction
from payments_api.schemas import PaymentDecision, PaymentRecord
from payments_api.service import PaymentService

router = APIRouter()


def get_service(request: Request) -> PaymentService:
    service: PaymentService = request.app.state.service
    return service


Service = Annotated[PaymentService, Depends(get_service)]


@router.post(
    "/payments",
    status_code=status.HTTP_201_CREATED,
    responses={200: {"description": "Duplicate transaction; the stored decision is returned"}},
)
async def create_payment(txn: Transaction, response: Response, service: Service) -> PaymentDecision:
    outcome = await service.submit(txn)
    if not outcome.created:
        response.status_code = status.HTTP_200_OK
    return PaymentDecision.from_row(outcome.payment)


@router.get("/payments/{payment_id}")
async def get_payment(payment_id: uuid.UUID, service: Service) -> PaymentRecord:
    payment = await service.get(payment_id)
    if payment is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "payment not found")
    return PaymentRecord.from_row(payment)


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request, service: Service, response: Response) -> dict[str, str]:
    """Database must answer. An unavailable scorer only fails readiness when
    REQUIRE_SCORER is set; otherwise requests are served by the rule fallback."""
    checks: dict[str, str] = {}
    healthy = True
    try:
        await service.ping_db()
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {type(exc).__name__}"
        healthy = False
    if await service.scorer.ready():
        checks["scorer"] = "ok"
    elif request.app.state.require_scorer:
        checks["scorer"] = "unavailable"
        healthy = False
    else:
        checks["scorer"] = "fallback"
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return checks


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
