from src.agent.graph import run_agent
result = run_agent('What is Delta Lake?')
print('Route:', result['route'])
print('Answer:', result['answer'][:200])

# Question dbt
result = run_agent("How do dbt models work?")
print(result["answer"])

# Question Spark
result = run_agent("What is Spark structured streaming?")
print(result["answer"])

# Question hors-sujet → doit router en 'direct'
result = run_agent("What is the capital of France?")
print("Route:", result["route"])  # doit afficher 'direct'