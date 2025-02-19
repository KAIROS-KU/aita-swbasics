from openai import OpenAI
from pydantic import BaseModel
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

def tail_question_process(prev_question, prev_ground_knowledge, prev_answer, current_question):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                너는 AI 조교 챗봇이야. 
                [이전 질문], [이전 배경 지식], [이전 답변], [현재 질문]을 기반으로 새로운 답변을 생성해."""
            },
            {
                "role": "user",
                "content": f"""
                [이전 질문]: {prev_question},
                [이전 배경 지식]: {prev_ground_knowledge},
                [이전 답변]: {current_question},
                [현재 질문]: {current_question}
                """
            }
        ]
    )
    tail_answer = completion.choices[0].message.content
    return tail_answer