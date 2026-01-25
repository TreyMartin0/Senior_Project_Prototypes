import json
import pandas as pd

# Load raw JSON files
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

audiogram = pd.DataFrame(load_json("raw_data/audiogram.json"))
audiogram_line = pd.DataFrame(load_json("raw_data/audiogram_line.json"))
audiogram_point = pd.DataFrame(load_json("raw_data/audiogram_point.json"))

#join audiogram points to lines
df = audiogram_point.merge(audiogram_line[["audiogramlineid", "audiogramid", "side", "transducertype", "type"]],
    on="audiogramlineid",
    how="left"
)

#join previous to audiogram
df = df.merge(audiogram[["audiogramid", "clientid", "date"]],
    on="audiogramid",
    how="left"
)

#filtering the data needed for prototype
df = df[(df["transducertype"] == "ac") &(df["type"] == "htl")]
df = df[["clientid", "audiogramid", "date", "side", "frequency", "level"]]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df.rename(columns={'clientid': 'userid'}, inplace=True)


# Define the output file path
output_file_path = 'test_data.json'

# Output the DataFrame to a JSON file
