from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Define the actions for bulk indexing
actions = [
    {
        "_op_type": "index",
        "_index": "plants",
        "_id": 2,
        "_source": {"name": "Tulip", "type": "Flower", "price": 18.0}
    },
    {
        "_op_type": "index",
        "_index": "plants",
        "_id": 3,
        "_source": {"name": "Cactus", "type": "Succulent", "price": 15.0}  # Document content
    }
]


success, failed = bulk(es, actions)

if success:
    print(f"Successfully indexed {success} documents.")
if failed:
    print(f"Failed to index {failed} documents.")

query = {
    "query": {
        "match_all": {}
    }
}

result = es.search(index="plants", body=query)
for hit in result['hits']['hits']:
    print(hit["_source"])
