import pandas as pd
import weaviate
from dateutil.parser import parse

def row_to_dict(row):
    # Convert the date to RFC3339 format
    date = parse(row['date']).isoformat() + "Z"

    # Create a data object
    return {
        "date": date,
        "open": row['open'],
        "high": row['high'],
        "low": row['low'],
        "close": row['close'],
        "volume": row['volume'],
        "openInt": row['openInt'],
        "combined": f"{date} {row['open']} {row['high']} {row['low']} {row['close']} {row['volume']} {row['openInt']}"
    }

# Load the CSV file
df = pd.read_csv('ibm.us.csv')

# Apply the transformation to each row
data_objects = df.apply(row_to_dict, axis=1).tolist()

# Initialize Weaviate client
client = weaviate.Client("http://localhost:8080")

# Prepare a batch process
with client.batch as batch:
    batch.batch_size = 100  # Set the batch size

    # Add the data objects to the batch
    for i, data_object in enumerate(data_objects):
        try:
            batch.add_data_object(data_object, "StockData")
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"Added {i + 1} data objects to the batch")
        except Exception as e:
            print(f"Failed to add data object at index {i}: {e}")

# Print completion message
print("Data objects addition process completed")
