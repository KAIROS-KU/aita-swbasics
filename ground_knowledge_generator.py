from openai import OpenAI
from supabase import create_client, Client
from pydantic import BaseModel
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)
supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_API_KEY)

class KnowledgeClassifyResponse(BaseModel):
    classification: str

def question_type_classifier(key_knowledge):
    supabase_response = supabase.table("DATA_TYPE").select("id", "data_type").execute()
    data_type_dict = {row["data_type"]: row["id"] for row in supabase_response.data}
    completion_1 = client.beta.chat.completions.parse(
        model="gpt-4o",
        response_format=KnowledgeClassifyResponse,
        messages=[
            {
                "role": "system", 
                "content": f"""
                데이터묶음 목록을 보고 user가 입력한 key_knowledge의 내용을 해결하기 위해서 목록 중 어디에 가장 부합하는지 딱 한가지만 골라.
                만약 그 어디에도 해당할거 같지 않다면 "해당없음"을 답변해.

                [데이터묶음 목록]: {list(data_type_dict.keys())}

                ex) 파이썬에서 리스트랑 튜플 차이가 뭐야?
                <response>: "코딩"
                """
            },
            {
                "role": "user", 
                "content": f"key_knowledge: {key_knowledge}"
            }
        ]
    )
    knowledge_classification = completion_1.choices[0].message.parsed.classification
    return knowledge_classification

def ground_knowledge_generator(key_knowledge):
    supabase_response = supabase.table("DATA_TYPE").select("id", "data_type").execute()
    data_type_dict = {row["data_type"]: row["id"] for row in supabase_response.data}
    knowledge_classification = question_type_classifier(key_knowledge)

    if knowledge_classification in data_type_dict:
        if knowledge_classification == "코딩":
            ground_knowledge = "코딩"
            return ground_knowledge
        data_type_id = data_type_dict[knowledge_classification]
        raw_data_response = supabase.table("RAW_DATA").select("data").eq("type_id", data_type_id).execute()
        raw_data_list = [row["data"] for row in raw_data_response.data]
        combined_text = "\n".join(raw_data_list)
        completion_2 = client.chat.completions.create(
            model="o1-preview",
            messages=[
                {
                    "role": "user",
                    "content": f"""
[Knowledge Base]: {combined_text}
- - -
[Knowledge Base]를 기반으로, 너가 [핵심 지식]에 적절히 매칭되는 지식을 가져와바. 만약 적절하게 매칭되는 지식이 없다면, "해당없음"으로 대답해줘.

- - -
[핵심 지식]: {key_knowledge}
"""
                }
            ]
        )
        ground_knowledge = completion_2.choices[0].message.content
    else:
        ground_knowledge = "해당없음"
    return ground_knowledge
