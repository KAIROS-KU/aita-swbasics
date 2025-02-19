from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

def answer_generator(question, ground_knowledge):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system", 
                "content": f"""
                [기반지식]을 기반으로 [수강생 문의]에 대응하는 적절한 답변을 구성해줘.

                [기반지식]: {ground_knowledge}
                """
            },
            {
                "role": "user", 
                "content": f"[수강생 문의]: {question}"
            }
        ]
    )
    answer = completion.choices[0].message.content
    return answer