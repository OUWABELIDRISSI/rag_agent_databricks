from src.agent.graph import run_agent
result = run_agent('What is Delta Lake?')
print('Route:', result['route'])
print('Answer:', result['answer'][:200])