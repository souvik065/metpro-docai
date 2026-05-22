"""
MacPro AI — LLM Answer Synthesis.

Takes retrieved sources + query → concise answer via LLM.

Supports:
- Anthropic Claude (default)
- OpenAI GPT-4
- Ollama (local, any model)

Swap providers by setting LLM_PROVIDER in .env.
"""
from __future__ import annotations

import json
from typing import Optional

from config.settings import settings
from src.models.schema import SourceReference
from src.utils.helpers import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are MacPro AI, a medical document assistant.
You are given a user query and retrieved context from medical documents.
Your job: answer the query concisely using ONLY the provided context.
If the answer is not in the context, say "Not found in the indexed documents."
Always reference which document/page supports your answer.
Do not fabricate medical information. Be precise and clinical in tone."""


class LLMSynthesizer:

    def synthesize(self, query: str, sources: list[SourceReference]) -> str:
        """Generate a concise answer from retrieved sources."""
        if not sources:
            return "No relevant documents found for your query."

        context = self._build_context(sources)
        prompt = f"Query: {query}\n\nContext:\n{context}"

        provider = settings.llm_provider.lower()
        try:
            if provider == "anthropic":
                return self._call_anthropic(prompt)
            elif provider == "openai":
                return self._call_openai(prompt)
            elif provider == "ollama":
                return self._call_ollama(prompt)
            elif provider == "azure_openai":
                return self._call_azure_openai(prompt)
            else:
                logger.warning(f"Unknown LLM provider: {provider}")
                return self._fallback_answer(sources)
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return self._fallback_answer(sources)

    def _build_context(self, sources: list[SourceReference]) -> str:
        parts: list[str] = []
        for i, src in enumerate(sources[:8], 1):  # cap at 8 sources
            snippet = src.snippet or ""
            loc = f"{src.filename} (page {src.page})" if src.page else src.filename
            parts.append(f"[{i}] {src.type.value.upper()} from {loc}:\n{snippet}")
        return "\n\n".join(parts)

    def _fallback_answer(self, sources: list[SourceReference]) -> str:
        """Simple extraction when LLM is unavailable."""
        top = sources[0]
        loc = f"page {top.page} of {top.filename}" if top.page else top.filename
        snippet = top.snippet or "(see source)"
        return f"Most relevant result found in {loc}: {snippet[:500]}"

    # ── Provider implementations ──────────────────────────────────────────

    def _call_anthropic(self, prompt: str) -> str:
        import anthropic
        # Use anthropic_api_key (will add to settings)
        api_key = getattr(settings, "anthropic_api_key", None) or settings.openai_api_key
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _call_azure_openai(self, prompt: str) -> str:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version="2024-12-01-preview",
            azure_endpoint=getattr(settings, "azure_openai_endpoint", ""),
        )
        response = client.chat.completions.create(
            model=settings.llm_model,  # Azure deployment name (not model name)
            max_tokens=settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def _call_openai(self, prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def _call_ollama(self, prompt: str) -> str:
        import httpx
        payload = {
            "model": settings.llm_model,
            "prompt": f"{_SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
        }
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
