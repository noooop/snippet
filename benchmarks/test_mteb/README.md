# numerical stability


Most models exhibit excellent numerical stability

| model name                              | st_main_score      | Difference             | std                    |
|-----------------------------------------|--------------------|------------------------|------------------------| 
| BAAI/bge-m3                             | 0.7873424632849964 | -4.014647728589615e-06 | 1.266416587059263e-05  | 
| BAAI/bge-base-en-v1.5                   | 0.7802846624612514 | 1.1294266950234721e-05 | 6.865350381034025e-06  | 
| Snowflake/snowflake-arctic-embed-xs     | 0.7149276890717804 | 1.5002530987628937e-05 | 5.132361246049283e-06  | 
| Snowflake/snowflake-arctic-embed-s      | 0.7408120447186094 | 1.2957674633273797e-05 | 5.364178900440517e-06  | 
| Snowflake/snowflake-arctic-embed-m      | 0.6467522411844727 | -3.727433978584216e-06 | 8.904071772230203e-06  | 
| Snowflake/snowflake-arctic-embed-l      | 0.6362746289758823 | 9.515755331335196e-05  | 2.023830079795977e-05  | 
| Snowflake/snowflake-arctic-embed-m-v1.5 | 0.6490882209298032 | 1.8871733633019083e-05 | 6.591107037250243e-06  | 
| Snowflake/snowflake-arctic-embed-l-v2.0 | 0.7122583106737259 | 1.074976228776503e-05  | 1.3400689624215418e-05 | 
| Snowflake/snowflake-arctic-embed-m-v2.0 | 0.7066229164460937 | 1.5418442692483048e-05 | 9.792523972420118e-06  | 
| Alibaba-NLP/gte-Qwen2-1.5B-instruct     | 0.7280529229028553 | 5.124313459714536e-05  | 1.6385524234026275e-05 | 

- intfloat/multilingual-e5-small shows a significant drop when using fp16 , and fp32 needs to be used. 

fp16: 
| intfloat/multilingual-e5-small | 0.7805425596252846 | -0.2749311085815237 | 0.006216913108536066 | 

fp32:
| intfloat/multilingual-e5-small | 0.7805425596252846 | -1.6403316041024851e-06 | 7.53539269543218e-06 | 

-  intfloat/multilingual-e5-large-instruct shows a significant drop when using fp16 , and fp32 needs to be used.

pooling_type="MEAN" + fp16 (default)
intfloat/multilingual-e5-large-instruct 0.8224491209469045 -0.28623335791513993 0.007169234312147499

pooling_type="MEAN" + fp32
intfloat/multilingual-e5-large-instruct 0.8224491209469045 -2.3497119421289625e-06 7.898194995699927e-06

- jinaai/jina-embeddings-v3 shows a slight drop when using fp16.

fp16:
| jinaai/jina-embeddings-v3 | 0.7834129787836271 | -0.0709833671361465 | 0.004834963031278825 | 
fp32:
| jinaai/jina-embeddings-v3 | 0.8243646209061513 | -3.119267999662778e-05 | 6.651161140301139e-06 |