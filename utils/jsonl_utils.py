import os
import gzip
import orjson
from collections import defaultdict
from config import logging

def load_jsonl_gz(file_path, single_object=True):
    logging.info(f"Loading JSONL file: {file_path}")
    data = defaultdict(list)
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        return data
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            if single_object:
                data.update(orjson.loads(f.read()))
            else:
                for line in f:
                    item = orjson.loads(line)
                    data.update(item)
        logging.info(f"Successfully loaded JSONL file: {file_path}")
    except Exception as e:
        logging.error(f"Error loading JSONL file: {file_path}, Error: {e}")
    return data

def save_jsonl_gz(file_path, data, single_object=True):
    logging.info(f"Saving JSONL file: {file_path}")
    try:
        with gzip.open(file_path, "wb") as f:
            if single_object:
                f.write(orjson.dumps(data))
            else:
                for key, value in data.items():
                    f.write(orjson.dumps({key: value}) + b"\n")
        logging.info(f"Successfully saved JSONL file: {file_path}")
    except Exception as e:
        logging.error(f"Error saving JSONL file: {file_path}, Error: {e}")

def extract_jsonl_gz(input_file_path, output_file_path):
    logging.info(f"Extracting JSONL file: {input_file_path} to {output_file_path}")
    try:
        with gzip.open(input_file_path, "rb") as infile, open(output_file_path, "w", encoding="utf-8") as outfile:
            for line in infile:
                data = orjson.loads(line)
                json_string = orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8")
                outfile.write(json_string + "\n")
        logging.info(f"Successfully extracted JSONL file: {input_file_path} to {output_file_path}")
    except Exception as e:
        logging.error(f"Error extracting JSONL file: {input_file_path} to {output_file_path}, Error: {e}")