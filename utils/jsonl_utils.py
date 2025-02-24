import os
import gzip
import orjson
from collections import defaultdict

def load_jsonl_gz(file_path, single_object=False):
    data = defaultdict(list)
    if not os.path.exists(file_path):
        return data
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        if single_object:
            data.update(orjson.loads(f.read()))
        else:
            for line in f:
                item = orjson.loads(line)
                data.update(item)
    return data

def save_jsonl_gz(file_path, data, single_object=False):
    with gzip.open(file_path, "wb") as f:
        if single_object:
            f.write(orjson.dumps(data))
        else:
            for key, value in data.items():
                f.write(orjson.dumps({key: value}) + b"\n")

def extract_jsonl_gz(input_file_path, output_file_path):
    with gzip.open(input_file_path, "rb") as infile, open(output_file_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = orjson.loads(line)
            
            json_string = orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8")
            
            outfile.write(json_string + "\n")
