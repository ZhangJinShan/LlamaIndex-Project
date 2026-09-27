from llama_index.readers.json import JSONReader

# 如果想保留原有的json格式，需要设置clean_json=False
reader = JSONReader(clean_json=False)
documents = reader.load_data(input_file="./data/request.json")
print(documents)

