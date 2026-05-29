# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import requests
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as OTLPGrpcExporter,
)

resource_attrs = {
    "service.name": "dummy_client",
}

endpoint = "http://localhost:4317"
resource = Resource.create(resource_attrs)
trace_provider = TracerProvider(resource=resource)
exporter = OTLPGrpcExporter(endpoint=endpoint, insecure=True)
trace_provider.add_span_processor(BatchSpanProcessor(exporter))
tracer = trace_provider.get_tracer("dummy_client")

url = "http://localhost:8000/v1/embeddings"
prompt = "San Francisco is a"

for i in range(10):
    payload = {
        "model": "BAAI/bge-base-en-v1.5",
        "input": prompt,
    }
    response = requests.post(url, json=payload)


with tracer.start_as_current_span("client-span", kind=SpanKind.CLIENT) as span:
    span.set_attribute("prompt", prompt)
    headers = {}
    TraceContextTextMapPropagator().inject(headers)
    payload = {
        "model": "BAAI/bge-base-en-v1.5",
        "input": prompt,
    }
    requests.post(url, headers=headers, json=payload)
