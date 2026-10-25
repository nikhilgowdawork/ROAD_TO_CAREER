
from research_agent.tools.websearch import search_web

result = search_web.invoke({
    "query": "Machine learning engineer skills required in India 2026"
})

print(f"Results found: {result['result_count']}")

for item in result["results"]:
    print(f"\nTitle: {item['title']}")
    print(f"URL: {item['url']}")
    print(f"Content: {item['content'][:300]}")
