import weaviate

# Create a client instance
client = weaviate.Client("http://localhost:8080")

# Get the first 10 data objects
objects = client.data_object.get(limit=10)

# Print out the retrieved data objects
for obj in objects['objects']:
    print(obj)