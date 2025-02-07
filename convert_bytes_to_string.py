import gzip
import orjson

CHUNK_INDEX = 0

def convert_bytes_to_string(input_file, output_file):
    with gzip.open(input_file, "rb") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = orjson.loads(line)
            
            json_string = orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8")
            
            outfile.write(json_string + "\n")

convert_bytes_to_string(f"{CHUNK_INDEX}_palm_landmarks.jsonl.gz", f"{CHUNK_INDEX}_str_palm_landmarks.jsonl")
convert_bytes_to_string(f"{CHUNK_INDEX}_body_landmarks.jsonl.gz", f"{CHUNK_INDEX}_str_body_landmarks.jsonl")
