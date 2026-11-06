"""OpenTelemetry traces of every model call, tool and graph node, exported over OTLP/HTTP.

Works with any collector that understands OpenInference spans (Phoenix, Langfuse, Grafana Tempo...).
Prompts and outputs stay out of the spans unless TRACE_CONTENT=true.
"""

import os

from ..config import Settings


def setup_tracing(settings: Settings) -> bool:
    if not settings.otlp_endpoint:
        return False
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise RuntimeError("Tracing needs the optional extra: uv sync --extra tracing") from exc
    if not settings.trace_content:
        for var in ("OPENINFERENCE_HIDE_INPUTS", "OPENINFERENCE_HIDE_OUTPUTS"):
            os.environ.setdefault(var, "true")
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": settings.service_name, "openinference.project.name": settings.service_name}
        )
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint)))
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    return True
