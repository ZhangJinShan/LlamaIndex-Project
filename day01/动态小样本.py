from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.schema import TextNode

from util.load_model import get_llm, get_embed

# 关键点：Settings.embed_model / Settings.llm 必须在建索引【之前】设置好，
# 否则 VectorStoreIndex 会回落到默认的 OpenAI embedding 并报 No API key
llm = get_llm()
embed = get_embed()

text_to_sql_prompt_tmpl_str = """
    你是一个故事生成专家
    下面是一些例子：
    
    示例：
    {{ examples }}
    
    
    现在轮到你了.
    问题: {{ query_str }}
    答案: 
"""

# 添加几个文档
example_nodes = [
    TextNode(
        text="Query: 请生成一个小红帽的故事，输出20字符\n小红帽去看奶奶，遇到大灰狼，被骗了，最后猎人救了她们。"
    ),
    TextNode(
        text="Query: 请生成一个白雪公主的故事，输出20字符\n白雪公主被后妈害，七矮人救她，王子吻醒了她。"
    ),
]
# 创建索引
index = VectorStoreIndex(nodes=example_nodes)

# 创建检索器
retriever = index.as_retriever(similarity_top_k=10)


def get_examples_fn(**kwargs):
    query = kwargs["query_str"]
    # 将用户的问题获取到之后，通过检索器从索引中去查询对应的示例
    examples = retriever.retrieve(query)
    return "\n\n".join(node.text for node in examples)

# 使用函数映射到提示模板中，会使用检索器找寻对应的样例填充到提示模板里的examples中
prompt_tmpl = RichPromptTemplate(
    text_to_sql_prompt_tmpl_str,
    function_mappings={"examples": get_examples_fn},
)

# 组装问题到提示词中
prompt = prompt_tmpl.format(
    query_str="请生成一个黑猫警长的故事"
)
print("prompt=>", prompt)

# 使用大模型进行回答
response = llm.complete(prompt)
print("response=>", response.text)
