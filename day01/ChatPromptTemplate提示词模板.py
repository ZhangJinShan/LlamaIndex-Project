from llama_index.core import ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole

message_templates = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="你是一个智能助手。"
    ),
    ChatMessage(
        role=MessageRole.USER,
        content="帮我生成一个关于{topic}的故事。"
    )
]

chat_prompt_template = ChatPromptTemplate(message_templates=message_templates)

# 格式化为聊天消息列表
messages = chat_prompt_template.format_messages(topic="狮子")
print(messages)
# 格式化为字符串
prompt = chat_prompt_template.format(topic="老虎")
print(prompt)
