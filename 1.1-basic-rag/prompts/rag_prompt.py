"""文档问答提示词模板。"""

from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT_TEMPLATE = """You are a helpful assistant.
Answer the question based only on the following context:
{context}

Question: {question}

Use the provided context to answer the user's question accurately and concisely.
Don't justify your answers.
Don't give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
