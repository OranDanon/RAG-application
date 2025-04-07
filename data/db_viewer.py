import lancedb

# Open your LanceDB connection
db = lancedb.connect("lancedb")

# Open the table you want
table = db.open_table("lang_graph_all-mpnet-base-v2")

# Convert to pandas DataFrame
df = table.to_pandas()

print(df.head(10))