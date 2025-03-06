from langchain.vectorstores.neo4j_vector import Neo4jVector
import logging
from openai import OpenAI
from neo4j import GraphDatabase
import config
from langchain_neo4j import Neo4jVector, Neo4jGraph, GraphCypherQAChain
from openai import OpenAI
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder
import time
from langchain_core.callbacks import BaseCallbackHandler


client = OpenAI(api_key=config.OPENAI_API_KEY)
neo4j_driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=config.OPENAI_API_KEY)
graph = Neo4jGraph(url=config.NEO4J_URI, username=config.NEO4J_USER, password=config.NEO4J_PASSWORD, refresh_schema=True, sanitize=True)

llm = ChatOpenAI(model="gpt-4o", api_key=config.OPENAI_API_KEY)

neo4j_vector = Neo4jVector.from_existing_graph(
    embedding=embedding_function,
    index_name="vector_index",
    retrieval_query="MATCH (n) WHERE n.embedding IS NOT NULL RETURN n",
    graph=graph,
    search_type="similarity",
    node_label="Chunk",
    embedding_node_property="embedding",
    text_node_properties=["name", "description"]
)

# ✅ GraphCypherQAChain (Cypher 기반 QA)
graph_qa_chain = GraphCypherQAChain.from_llm(
    cypher_llm=llm,
    qa_llm=llm,
    validate_cypher=True,
    graph=graph,
    return_intermediate_steps=True,
    top_k=3,
    allow_dangerous_requests=True
)

# def get_graph_response(question):
#     try:
#         response = graph_qa_chain.run(question)
#         return {"response": response, "mode": "graph"}
#     except Exception as e:
#         logging.error(f"Error in graph response: {e}")
#         return {"response": "An error occurred.", "mode": "graph"}
    
# def get_rag_response(question):
#     try:
#         docs = neo4j_vector.similarity_search(question, k=3)
#         context = "\n\n".join([doc.page_content for doc in docs])

#         ai_response = llm.invoke(f"Context: {context}\n\nQuestion: {question}")
#         return {"response": ai_response, "context": context, "mode": "rag"}
#     except Exception as e:
#         logging.error(f"Error retrieving RAG response: {e}")
#         return {"response": "An error occurred.", "mode": "rag"}

# def chat_with_graph(question, mode="graph"):
#     if mode == "graph":
#         return get_graph_response(question)
#     elif mode == "rag":
#         return get_rag_response(question)
#     else:
#         return {"response": "Invalid mode."}

# response_graph = chat_with_graph("리스트와 튜플의 차이가 뭐야?", mode="rag")
# print(response_graph)

# ✅ 후속 질문을 생성하는 LangChain 프롬프트
follow_up_prompt = ChatPromptTemplate.from_template(
    """
    사용자의 질문: {prev_question}

    학생이 이 개념을 더 깊이 이해할 수 있도록 적절한 후속 질문을 하나만 만들어 주세요.

    **강의 자료속 내용이면 좋음.
    **사용자의 질문 속 개념과 유사하나 다른 개념에 관한 질문.
    **"~~에 대해 자세히 설명해줘" 식의 말투.
    **파이썬 코딩과 관련된 개념이어야 해.
    **사족 붙이지 말고 질문 하나만 답변해.

    ex) 딕셔너리에 대해 자세히 설명해줘.
    ex2) 파이썬에서 데이터 형 변환하는 법을 자세히 설명해줘.
    ex3) 리스트와 튜플의 각각의 장단점을 자세하게 설명해줘.
    """
)

# ✅ LangChain LLMChain 생성
follow_up_chain = LLMChain(llm=llm, prompt=follow_up_prompt)

def follow_up_question_generator(prev_question):
    graph_result = graph_qa_chain.run(prev_question)
    follow_up_question = follow_up_chain.run(prev_question=prev_question, context=graph_result)
    return follow_up_question.strip()



