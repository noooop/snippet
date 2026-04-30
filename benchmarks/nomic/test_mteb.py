import mteb
from sentence_transformers import SentenceTransformer


model_name = "nomic-ai/nomic-embed-text-v1"
revision="0759316f275aa0cb93a5b830973843ca66babcf5"


model = mteb.get_model(model_name, revision=revision)
tasks = mteb.get_tasks(tasks=["STS12"])
results = mteb.evaluate(model, tasks=tasks)
main_score = results[0].scores["test"][0]["main_score"]

print(main_score)
