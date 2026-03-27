import json
import os
import re
import uuid
import asyncio
from urllib.parse import urlencode
from typing import Any, Dict, List, Optional, AsyncGenerator

import httpx
import snowflake.connector
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.json_log import json_log
from app.db.snowflake_pool import get_snowflake_pool
from .intent_router.intent_detector import IntentDetector
from .intent_router.models import IntentResult, RegistryData
from app.routes.analyst import AnalystRequest, analyst_ask_analyst

router = APIRouter(prefix="/ai", tags=["ai-test"])

DATA_API_BASE_URL = os.getenv("DATA_API_BASE_URL", "http://localhost:8080/titan-data-products")


class IntentDetectionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    workflow: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)
    session_context: Optional[dict] = None


class IntentDetectionResponse(BaseModel):
    success: bool
    intent_type: str
    route: Optional[dict] = None
    parameters: Optional[dict] = None
    confidence: Optional[float] = None
    detection_method: str
    metadata: Optional[dict] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    data_api_base_url: str
    model: str
    correlation_id: str


def _normalize_workflow_key(workflow: Optional[str]) -> str:
    return (workflow or "").strip().upper()


def _build_resolved_route_path(route: Optional[dict], params: Optional[dict]) -> Optional[str]:
    if not route or not isinstance(route, dict):
        return None
    path = route.get("path")
    if not path or not isinstance(params, dict):
        return path
    method = (route.get("method") or "GET").upper()
    if method == "POST":
        return path
    try:
        resolved_path, remaining = _apply_path_params(path, params)
        if remaining:
            query = urlencode(remaining, doseq=True)
            return f"{resolved_path}?{query}" if query else resolved_path
        return resolved_path
    except KeyError:
        base_path = _strip_path_params(path)
        query = urlencode(params, doseq=True)
        return f"{base_path}?{query}" if query else base_path


def _build_resolved_route_body(route: Optional[dict], params: Optional[dict]) -> Optional[dict]:
    if not route or not isinstance(route, dict):
        return None
    if not isinstance(params, dict):
        return None
    method = (route.get("method") or "GET").upper()
    if method != "POST":
        return None
    return params


def _apply_path_params(path: str, params: dict) -> tuple[str, dict]:
    remaining = dict(params)

    def replace_match(match):
        key = match.group(1)
        if key not in remaining:
            raise KeyError(f"Missing path parameter: {key}")
        value = remaining.pop(key)
        # Handle list vs scalar for path substitution
        if isinstance(value, list) and value:
            # Join multiple values with commas or handle as repeated keys? 
            # Most dashboard APIs for single path segments use comma-separated lists 
            # or we can use the repeated key strategy if that's standard for this API.
            # Based on common REST patterns for "in-path" filtering:
            val_str = ",".join(str(v) for v in value)
        else:
            val_str = str(value)
            
        # Return in key=value format for path segments
        return f"{key}={val_str}"

    new_path = re.sub(r"\{([^}]+)\}", replace_match, path)

    # For remains, ensure we also flatten single-element lists for query parameters
    # if the route definition doesn't explicitly allow multi-values.
    # We do a shallow normalization here for the return.
    for k, v in remaining.items():
        if isinstance(v, list) and len(v) == 1:
            # Check if this parameter is in the map/dashboard registry as wanting an array
            # If not, we flatten it for the action URL
            remaining[k] = v[0]

    return new_path, remaining


def _strip_path_params(path: str) -> str:
    if not path:
        return path
    return re.sub(r"/\{[^/]+\}", "", path)


def _select_semantic_view(prompt: str, semantic_views: List[Dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    if not semantic_views:
        return None, None
    lowered = (prompt or "").lower()
    prompt_tokens = {t for t in re.split(r"\W+", lowered) if len(t) > 2}
    best_score = -1
    best_name = None
    best_reason = None
    for view in semantic_views:
        tokens = set()
        for value in (
            view.get("name"),
            view.get("description"),
            view.get("intent"),
        ):
            for token in re.split(r"\W+", str(value or "")):
                token = token.strip().lower()
                if len(token) > 2:
                    tokens.add(token)
        for utterance in (view.get("example_utterances") or []):
            for token in re.split(r"\W+", str(utterance or "")):
                token = token.strip().lower()
                if len(token) > 2:
                    tokens.add(token)
        matched = sorted(tokens & prompt_tokens)
        score = len(matched)
        if score > best_score:
            best_score = score
            best_name = view.get("name")
            best_reason = (
                f"Matched semantic view tokens {matched[:5]} using name/description/intent/example_utterances."
                if matched
                else "Defaulted to the first semantic view due to no token matches."
            )
    if not best_name:
        return semantic_views[0].get("name"), "Defaulted to the first semantic view due to no token matches."
    return best_name, best_reason



async def _fetch_data_api_registry(base_url: str) -> dict:
    url = f"{base_url.rstrip('/')}/route-registry"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach data-api registry at {url}: {exc}",
        ) from exc
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to load data-api registry: {response.status_code} {response.text}",
        )
    return response.json()


def _build_registry_from_data_api(data_registry: dict, region: Optional[str] = None) -> dict:
    regions = data_registry.get("regions", {}) or {}
    global_endpoints = data_registry.get("global_endpoints", []) or []
    if not regions:
        return {}

    region_key_map = {str(key).strip().upper(): key for key in regions.keys()}

    if region:
        region_key = region.strip().upper()
        resolved_key = region_key_map.get(region_key)
        if resolved_key is None:
            raise HTTPException(status_code=404, detail=f"Unknown region: {region}")
        preferred_region = resolved_key
        regions = {preferred_region: regions.get(preferred_region, {})}
    else:
        preferred_region = region_key_map.get("INTL") or next(iter(regions.keys()))

    region_entry = regions.get(preferred_region, {}) or {}
    workflows_entry = region_entry.get("workflows", {}) or {}

    workflows: Dict[str, Dict[str, Any]] = {}
    for workflow_name, workflow_config in workflows_entry.items():
        workflow_key = _normalize_workflow_key(workflow_name)
        routes: List[Dict[str, Any]] = []

        for route in workflow_config.get("dashboard_routes", []) or []:
            route_entry = dict(route)
            route_entry.setdefault("intent", "dashboard_load")
            routes.append(route_entry)

        for endpoint in global_endpoints:
            endpoint_entry = dict(endpoint)
            endpoint_name = endpoint_entry.get("name")

            if endpoint_name == "ask_analyst":
                endpoint_entry.setdefault("method", "POST")
                endpoint_entry.setdefault("intent", "analytical_question")
                endpoint_entry.setdefault("response_type", "streaming_analytical")
            elif endpoint_name == "layers_viewport":
                endpoint_entry.setdefault("method", "GET")
                endpoint_entry.setdefault("intent", "map_update")
                endpoint_entry.setdefault("response_type", "map_update")
            elif endpoint_name == "layers_feature_ids":
                endpoint_entry.setdefault("method", "POST")
                endpoint_entry.setdefault("intent", "map_select")
                endpoint_entry.setdefault("response_type", "map_select")
            else:
                endpoint_entry.setdefault("method", "GET")

            routes.append(endpoint_entry)

        workflows[workflow_key] = {
            "semantic_views": workflow_config.get("semantic_views", []),
            "routes": routes,
            "map_layers": workflow_config.get("map_layers", []),
            "common_map_layers": region_entry.get("common_map_layers", []),
            "region": preferred_region,
        }

    return {
        "schema_version": data_registry.get("schema_version"),
        "regions": regions,
        "global_endpoints": global_endpoints,
        "workflows": workflows,
    }


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-test"}


@router.get("/registry")
async def get_registry(
    workflow: Optional[str] = Header(None),
    region: Optional[str] = Header(None),
    data_api_base_url: Optional[str] = Header(None),
):
    base_url = data_api_base_url or DATA_API_BASE_URL
    data_registry = await _fetch_data_api_registry(base_url)
    registry = _build_registry_from_data_api(data_registry, region=region)
    workflows = registry.get("workflows", {})

    if workflow:
        workflow_key = workflow.strip().upper()
        if workflow_key not in workflows:
            raise HTTPException(status_code=404, detail=f"Unknown workflow: {workflow}")
        registry = {**registry, "workflows": {workflow_key: workflows[workflow_key]}}

    return {
        **registry,
        "data_api_base_url": base_url,
        "request_headers": {"workflow": workflow, "region": region},
    }


@router.post("/intent/detect", response_model=IntentDetectionResponse)
async def detect_intent_llm(
    body: IntentDetectionRequest,
    data_api_base_url: Optional[str] = Header(None),
    fastapi_request: Request = None,
    pool=Depends(get_snowflake_pool),
):
    """
    Detect intent using the 3-step modular LLM detector.
    
    This endpoint uses the LLMDetector to:
    1. Classify user intent based on data-api capabilities
    2. Extract domain-specific entities and parameters 
    3. Validate and clean extracted parameters against registry
    """
    workflow = body.workflow
    region = body.region
    
    # Validate inputs
    if not workflow:
        raise HTTPException(status_code=400, detail="Missing workflow")
    if not region:
        raise HTTPException(status_code=400, detail="Missing region")
    if region.strip().upper() not in {"NA", "INTL"}:
        raise HTTPException(status_code=400, detail="Invalid region. Allowed values: NA, INTL")

    # Fetch and build registry
    base_url = data_api_base_url or DATA_API_BASE_URL
    try:
        data_registry = await _fetch_data_api_registry(base_url)
        registry = _build_registry_from_data_api(data_registry, region=region)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch registry: {str(e)}")

    workflows = registry.get("workflows", {})
    workflow_key = workflow.strip().upper()
    if workflow_key not in workflows:
        raise HTTPException(status_code=404, detail=f"Unknown workflow: {workflow}")

    # Get correlation ID
    correlation_id = fastapi_request.headers.get("x-correlation-id") if fastapi_request else None
    if not correlation_id:
        correlation_id = str(uuid.uuid4())

    # Create LLM detector and call detect_intent
    try:
        detector = ModularLLMDetector(pool=pool)
        
        # Convert built registry to RegistryData format
        registry_data = type('RegistryData', (), {})()
        workflow_registry = workflows[workflow_key]
        
        # Set registry attributes
        registry_data.routes = workflow_registry.get("routes", [])
        registry_data.semantic_views = workflow_registry.get("semantic_views", [])
        registry_data.map_layers = workflow_registry.get("map_layers", [])
        registry_data.common_map_layers = workflow_registry.get("common_map_layers", [])
        registry_data.global_endpoints = registry.get("global_endpoints", [])
        registry_data.regions = registry.get("regions", {})
        
        # Detect intent with workflow and region
        result = detector.detect_intent(
            prompt=body.prompt,
            registry=registry_data,
            workflow=workflow.lower(),
            region=region.lower(),
            session_context=body.session_context
        )
        
        # Convert IntentResult to response format
        return IntentDetectionResponse(
            success=result.success,
            intent_type=result.intent_type,
            route=result.route,
            parameters=result.parameters,
            confidence=result.confidence,
            detection_method=result.detection_method,
            metadata=result.metadata,
            error_message=result.error_message,
            error_type=result.error_type,
            data_api_base_url=base_url,
            model="llm_detector_3_step",
            correlation_id=correlation_id
        )
        
    except Exception as e:
        json_log(
            "Exception during LLM intent detection",
            event="ai-intent-detect",
            error=str(e),
            correlation_id=correlation_id
        )
        raise HTTPException(status_code=500, detail=f"Intent detection failed: {str(e)}")


@router.post("/intent/detect-stream")
async def detect_intent_stream(
    body: IntentDetectionRequest,
    data_api_base_url: Optional[str] = Header(None),
    fastapi_request: Request = None,
    pool=Depends(get_snowflake_pool),
):
    """
    Streaming version of intent detection that provides intermediate events:
    - intent: Initial intent classification and parameter extraction
    - action: Executable endpoint details for map/dashboard (GET/POST with params)
    - data: Response from /ask-analyst if applicable
    """
    workflow = body.workflow
    region = body.region
    correlation_id = fastapi_request.headers.get("x-correlation-id") or str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # 1. Initialization and Registry
            base_url = data_api_base_url or DATA_API_BASE_URL
            data_registry = await _fetch_data_api_registry(base_url)
            registry = _build_registry_from_data_api(data_registry, region=region)
            
            workflows = registry.get("workflows", {})
            workflow_key = workflow.strip().upper()
            if workflow_key not in workflows:
                yield f"data: {json.dumps({'event': 'error', 'message': f'Unknown workflow: {workflow}'})}\n\n"
                return

            # 2. Intent Detection
            detector = ModularLLMDetector(pool=pool)
            
            # Use the SAME object structure as detect_intent_llm to ensure consistency
            registry_data = type('RegistryData', (), {})()
            workflow_registry = workflows[workflow_key]
            
            # Set registry attributes precisely as in the working endpoint
            registry_data.routes = workflow_registry.get("routes", [])
            registry_data.semantic_views = workflow_registry.get("semantic_views", [])
            registry_data.map_layers = workflow_registry.get("map_layers", [])
            registry_data.common_map_layers = workflow_registry.get("common_map_layers", [])
            registry_data.global_endpoints = registry.get("global_endpoints", [])
            registry_data.regions = registry.get("regions", {})

            # run in threadpool as it might be blocking
            result: IntentResult = await run_in_threadpool(
                detector.detect_intent,
                prompt=body.prompt,
                registry=registry_data,
                workflow=workflow.lower(),
                region=region.lower(),
                session_context=body.session_context
            )

            # intent: Initial intent classification and parameter extraction
            intent_event = {
                "event": "intent",
                "success": result.success,
                "intent_type": result.intent_type,
                "confidence": result.confidence,
                "parameters": result.parameters,
                "metadata": result.metadata,
                "correlation_id": correlation_id
            }

            # If it's a dashboard_load or map_update, we should ensure the intent event parameters 
            # are flattened if they are single-element lists, to match the action event consistency.
            if result.success and result.parameters:
                flattened_params = dict(result.parameters)
                for k, v in flattened_params.items():
                    if isinstance(v, list) and len(v) == 1:
                        flattened_params[k] = v[0]
                intent_event["parameters"] = flattened_params

            yield f"data: {json.dumps(intent_event)}\n\n"

            if not result.success:
                return

            # 3. Action Event (Map/Dashboard only)
            # We skip action event for analytical/ask-analyst since it's executed internally
            # UNLESS a dashboard fallback was identified in metadata
            resolved_route = result.route
            is_analytical = result.intent_type == "analytical" or (resolved_route and resolved_route.get("name") == "ask_analyst")
            dashboard_action = None

            # Check for dashboard availability in metadata (even for analytical)
            metadata = result.metadata or {}
            dashboard_avail = metadata.get("_dashboard_availability")
            if dashboard_avail and isinstance(dashboard_avail, dict):
                routes_list = dashboard_avail.get("dashboard_routes") or []
                if routes_list and isinstance(routes_list, list):
                    # Pick the first satisfied dashboard
                    dashboard_action = routes_list[0]

            if (resolved_route and not is_analytical) or dashboard_action:
                # Use the dashboard route if it's an analytical fallback, otherwise the resolved route
                route_to_use = dashboard_action if dashboard_action else resolved_route
                
                # Extract parameters for action - use extracted_parameters if it's a dashboard fallback
                if dashboard_action and "extracted_parameters" in dashboard_action:
                    # For fallback dashboards in analytical intents, we ONLY want the extracted dashboard params
                    # plus standard region/workflow, avoiding the inclusion of analyst-specific params like 'question'
                    action_params = dict(dashboard_action["extracted_parameters"])
                    action_params.setdefault("workflow", result.parameters.get("workflow", workflow))
                    action_params.setdefault("region", result.parameters.get("region", region))
                else:
                    action_params = result.parameters

                # Create a local copy to avoid modifying the registry objects
                action_route = dict(route_to_use)
                path = _build_resolved_route_path(action_route, action_params)
                body_params = _build_resolved_route_body(action_route, action_params)
                method = (action_route.get("method") or "GET").upper()

                action_event = {
                    "event": "action",
                    "intent": result.intent_type,
                    "route_name": action_route.get("name"),
                    "method": method,
                    "path": path,
                    "body": body_params,
                    "data_api_base_url": base_url,
                    "correlation_id": correlation_id
                }
                yield f"data: {json.dumps(action_event)}\n\n"

            # 4. Data Event (only for analytical/ask-analyst)
            if is_analytical:
                # Always prioritize semantic_view from parameters if set by LLM/logic
                params = result.parameters or {}
                semantic_model = params.get("semantic_view") or params.get("semantic_model_name")
                print(f"Semantic model for ask-analyst: {semantic_model}")
                # If LLM returned a specific view, use it directly. 
                # Otherwise if it's generic, try the token-based fallback.
                if not semantic_model or semantic_model == "ask_analyst":
                    semantic_views = workflows[workflow_key].get("semantic_views", [])
                    res_tuple = _select_semantic_view(body.prompt, semantic_views)
                    if res_tuple and isinstance(res_tuple, tuple):
                        resolved_view = res_tuple[0]
                        if resolved_view:
                            semantic_model = resolved_view

                if semantic_model and semantic_model != "ask_analyst":
                    print(f"Executing ask-analyst via external API: {base_url}/ask-analyst")
                    try:
                        analyst_payload = {
                            "question": body.prompt,
                            "semantic_model_name": semantic_model,
                            "messages": (body.session_context.get("messages") if body.session_context else []) or []
                        }
                        
                        async with httpx.AsyncClient(timeout=60.0) as client:
                            analyst_url = f"{base_url.rstrip('/')}/ask-analyst"
                            resp = await client.post(
                                analyst_url,
                                json=analyst_payload,
                                headers={"x-correlation-id": correlation_id}
                            )
                            
                            if resp.status_code == 200:
                                data_content = resp.json()
                                yield f"data: {json.dumps({'event': 'data', 'content': data_content, 'correlation_id': correlation_id})}\n\n"
                            else:
                                error_detail = resp.text
                                yield f"data: {json.dumps({'event': 'error', 'message': f'Analyst API failed ({resp.status_code}): {error_detail}'})}\n\n"
                    except Exception as ae:
                        json_log("Streaming analyst error", error=str(ae), correlation_id=correlation_id)
                        yield f"data: {json.dumps({'event': 'error', 'message': f'Analyst connection failed: {str(ae)}'})}\n\n"

        except Exception as e:
            json_log("Streaming intent detection error", error=str(e), correlation_id=correlation_id)
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
