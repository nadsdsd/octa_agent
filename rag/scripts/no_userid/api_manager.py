from __future__ import annotations

from typing import Dict, List

import config


def get_service_candidates(service_name: str) -> List[Dict]:
    if service_name in getattr(config, "SERVICE_CANDIDATES", {}):
        return list(config.SERVICE_CANDIDATES[service_name])

    if service_name not in config.API_ENDPOINTS:
        raise KeyError(f"未知服务名: {service_name}")

    return [
        {
            "name": service_name,
            "base_url": config.TOOL_BACKEND_URL.rstrip("/"),
            "path": config.API_ENDPOINTS[service_name],
            "probe_url": f"{config.TOOL_BACKEND_URL.rstrip('/')}/health",
        }
    ]


def get_service_url(service_name: str) -> str:
    candidates = get_service_candidates(service_name)
    first = candidates[0]
    return f"{first['base_url']}{first['path']}"


def provider_is_available(provider_name: str) -> bool:
    provider = config.LLM_PROVIDERS.get(provider_name.upper())
    if not provider:
        return False

    normalized = provider_name.upper()
    if normalized in {"ZHIPU", "SILICONFLOW"} and not provider.get("api_key"):
        return False

    return bool(provider.get("enabled", True))


def get_provider_config(provider_name: str) -> Dict:
    normalized = provider_name.upper()
    if normalized not in config.LLM_PROVIDERS:
        raise KeyError(f"未知 LLM Provider: {provider_name}")
    return config.LLM_PROVIDERS[normalized]


def get_llm_provider_order(preferred: str | None = None) -> List[str]:
    primary = (preferred or config.DEFAULT_LLM_PROVIDER).upper()
    order = [primary, *config.LLM_FALLBACKS.get(primary, [])]

    deduped: List[str] = []
    for name in order:
        normalized = name.upper()
        if normalized not in deduped and provider_is_available(normalized):
            deduped.append(normalized)

    if not deduped:
        raise RuntimeError("没有可用的 LLM Provider，请检查 DEFAULT_LLM_PROVIDER / API KEY / 本地模型配置。")

    return deduped