# asyncio is really slow

PTAL [#11945](https://github.com/vllm-project/vllm/issues/11945)

## setting
- hardware 13700kf + ddr4 3600 128G

## run
```commandline
python -m benchmarks.dummy_zmq_loop.main
```

## result 
|                 | server: naive | server: gevent | server: asyncio | server: uvloop | avg all server | 
|-----------------|---------------|----------------|-----------------|----------------|----------------| 
| client: naive   | 53586.69      | 39127.87       | 34617.86        | 37090.49       | 41105.7275     | 
| client: gevent  | 38408.05      | 30665.28       | 26724.27        | 28110.91       | 30977.1275     | 
| client: asyncio | 34353.76      | 27077.97       | 23354.54        | 26413.4        | 27799.9175     | 
| client: uvloop  | 37912.84      | 30066.15       | 25288.72        | 28999.83       | 30566.885      | 
| avg all client  | 41065.335     | 31734.3175     | 27496.3475      | 30153.6575     |                |  


## conclusion
asyncio is really slow
