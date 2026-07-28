import asyncio
import json
import time

import aiohttp
import anyio
from prettytable import PrettyTable, TableStyle

api_url = "http://localhost:8000/score"


async def rerank(session, api_url, query:str, documents:list[str]):
    data = {
        "model": "bge-reranker-v2-m3",
        "queries": query,
        "documents": documents
    }
    dt1 = time.time_ns()
    async with session.post(api_url, json=data, ssl=False, timeout=None,
                            headers={"Connection": "close"}) as response:
        dt2 = time.time_ns()
        elps = (dt2 - dt1) // (1_000_000)
        if response.status == 200:
            res = await response.json()
        else:
            response.raise_for_status()
        return (elps, res)

async def t_rerank(data:dict, concurrency:int):
    connector = aiohttp.TCPConnector(ssl=False, limit=0, limit_per_host=50)

    query = data["query"]
    docs = data["docs"]
    slide_size = 100
    slides = [docs[i : i + slide_size] for i in range(0, len(docs), slide_size)]
    results = []
    limiter = anyio.CapacityLimiter(concurrency)
    async with aiohttp.ClientSession(connector=connector, timeout=None) as session:
        async def run_rerank(session, api_url, query, batch):
            res = await rerank(session, api_url, query, batch)
            results.append(res[0])
        async with limiter:
            async with anyio.create_task_group() as tg:
                for batch in slides:
                    tg.start_soon(run_rerank, session, api_url, query, batch)
    return round(max(results),2), round(min(results),2), round(sum(results)/len(results),2),len(results)



async def t_rerank_probe(data):
    table = PrettyTable()
    table.field_names =["Concurrency", "Avg msec", "Max msec", "Min msec"]
    table.align = "l"
    table.set_style(TableStyle.MARKDOWN)
    for c in [1,5,10,15,20,30, 50]:
        r_max, r_min, r_avg, calls = await t_rerank(data,c)
        #print(f"{c}\t\{r_avg},\t{r_max}\t{r_min}")
        table.add_row([c,  r_avg, r_max,r_min])
        print(table.get_string())


if __name__ == '__main__':
    with open("./_test_data/rerank.json", "r", encoding="utf-8") as f:
        data_str = f.read()
    data  = json.loads(data_str)
    asyncio.run(t_rerank_probe(data))
    #asyncio.run(t_sql())
    exit(0)