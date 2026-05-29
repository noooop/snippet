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

url = "http://localhost:8000/score"

model_name = "BAAI/bge-reranker-v2-m3"

get_input_len = lambda output_len: (output_len - 4) // 2


input_len = 512

_input_len = get_input_len(input_len)

queries = "你" * _input_len
documents = "你" * _input_len

payload = {"model": model_name, "queries": queries, "documents": documents}

for i in range(10):
    response = requests.post(url, json=payload)

with tracer.start_as_current_span("client-span", kind=SpanKind.CLIENT) as span:
    headers = {}
    TraceContextTextMapPropagator().inject(headers)
    requests.post(url, headers=headers, json=payload)
