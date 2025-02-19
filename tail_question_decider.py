from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

def tail_question_generator(prev_question, current_question):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """
                너는 AI 조교 챗봇이야. 
                [이전 질문]과 [현재 질문]을 보고 [현재 질문]이 [이전 질문]의 꼬리질문인지 아니면 새로운 질문인지를 판별해.
                
                <response format>
                답변은 무조건 yes나 no로만 답해.

                [현재 질문]이 [이전 질문]의 꼬리질문이면: yes
                [현재 질문]이 [이전 질문]의 꼬리질문이 아니면: no
                """
            },
            {
                "role": "user",
                "content": f"""
                [이전 질문]: {prev_question},
                [현재 질문]: {current_question}"""
            }
        ]
    )
    tail_question_response = completion.choices[0].message.content

    if "yes" in tail_question_response:
        tail_question_decide = "yes"
    elif "no" in tail_question_response:
        tail_question_decide = "no"
    else:
        tail_question_decide = "no"
        
    return tail_question_decide