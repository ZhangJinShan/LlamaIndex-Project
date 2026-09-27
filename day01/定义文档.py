from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader

text_list = ["text1", "text2"]
# 创建文档对象，并添加元数据
documents = [
    Document(text=t, metadata={"filename": "文件名称", "category": "类别"}) for t in text_list
]
print(documents)

# 自动设置元数据
def filename_fn(filename: str):
    return {
        "file_name": filename,
        "category": Path(filename).suffix,
    }


reader_documents = SimpleDirectoryReader("./data", file_metadata=filename_fn).load_data()
print(reader_documents)
