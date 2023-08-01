import weaviate

client = weaviate.Client('http://localhost:8080')

# Define the schema
# Define the schema
schema = {
    "classes": [
        {
            "class": "StockData",
            "vectorizer": "text2vec-openai",  # specify the vectorizer
            "properties": [
                {"name": "date", "dataType": ["date"]},
                {"name": "open", "dataType": ["number"]},
                {"name": "high", "dataType": ["number"]},
                {"name": "low", "dataType": ["number"]},
                {"name": "close", "dataType": ["number"]},
                {"name": "volume", "dataType": ["int"]},
                {"name": "openInt", "dataType": ["int"]},
                {"name": "combined", "dataType": ["text"]}
            ]
        }
    ]
}

# Create the schema
client.schema.create(schema)