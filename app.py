import streamlit as st
from openai import OpenAI
from supabase import create_client
import config
import key_knowledge_generator
import answer_generator
import ground_knowledge_generator
import tail_question_decider
import tail_question_process
import follow_up_question_generator
from predibase import Predibase
import random

# Supabase & OpenAI 클라이언트 초기화
supabase = create_client(config.SUPABASE_URL, config.SUPABASE_API_KEY)
client = OpenAI(api_key=config.OPENAI_API_KEY)
pb = Predibase(api_token=config.PREDIBASE_API_TOKEN)

predeibase_model_system_message = """
당신은 대학 신입생을 대상으로 한 파이썬 기초 수업의 조교(TA) 역할을 수행하는 AI입니다. 학생들이 쉽고 효과적으로 파이썬을 학습할 수 있도록 도와주세요.

1. 개념 설명
- 쉬운 언어로 설명하고, 어려운 용어는 피하세요.
- 직관적인 비유와 예제를 활용하여 개념을 설명하세요.
- 핵심 내용을 먼저 요약하고, 추가적인 설명을 단계적으로 제공하세요.

2. 문법 및 코드 예제 제공
- 문법 설명 시 간단한 코드 예제와 함께 제공하세요.
- 실행 결과를 함께 제시하여 학습자가 즉시 이해할 수 있도록 도와주세요.
- 필요할 경우 추가적인 설명을 제공하고, 코드 개선 방법도 안내하세요

3. 코드 오류 분석 및 디버깅 지원
- 오류 발생 시 다음 단계를 따르세요.
  1) 오류 분석: 오류 메시지를 해석하고 발생 원인을 설명하세요.
  2) 해결 방법: 수정된 코드와 실행 결과를 제공하고, 수정 이유를 설명하세요.
  3) 추가적으로 출력 형식이 알맞은지 (스페이스바, 오타, 엔터, 인덴테이션 등등)를 검사하라고 다시 한번 학생에게 언급해주세요.

4. 학습 지원 및 추가 기능
- 학생들이 연습할 수 있도록 관련된 학습 자료(튜토리얼, 연습 문제 등)를 제공하세요.
- 코드 스타일 개선 방법을 안내하세요.
- 학생이 부담 없이 질문할 수 있도록 친절한 톤을 유지하세요.
- 질문이 모호할 경우 추가 질문을 유도하여 학생이 원하는 답변을 얻을 수 있도록 도와주세요.

"""

def coding_question_process_one(txt):
    prompt = (
        f"<|im_start|>system\n{predeibase_model_system_message}<|im_end|>\n"
        f"<|im_start|>user\n{txt}<|im_end|>\n<|im_start|>assistant\n"
    )
    lorax_client = pb.deployments.client("solar-1-mini-chat-240612")
    coding_answer = lorax_client.generate(prompt, temperature=0.7, adapter_id=config.PREDIBASE_MODEL_ONE_ADAPT_ID).generated_text
    return coding_answer

def coding_question_process_two(txt):
    prompt = (
        f"<|im_start|>system\n{predeibase_model_system_message}<|im_end|>\n"
        f"<|im_start|>user\n{txt}<|im_end|>\n<|im_start|>assistant\n"
    )
    lorax_client = pb.deployments.client("solar-1-mini-chat-240612")
    coding_answer = lorax_client.generate(prompt, temperature=0.7, adapter_id=config.PREDIBASE_MODEL_TWO_ADAPT_ID).generated_text
    return coding_answer

def save_chat_to_db(user_question, ai_answer, is_main_question, question_type):
    data = {
        "question": user_question,
        "answer": ai_answer,
        "main_question": is_main_question,
        "question_type": question_type
    }
    supabase.table("CONSULTING_CHAT").insert(data).execute()
    return



# Streamlit UI 설정
st.set_page_config(page_title="AI 조교", page_icon="🤖")
st.title("🤖 AI 조교")

with st.sidebar:
    st.markdown("# **⚡ AI 조교 사용 안내**")
    
    st.info(
        """
        - **정확한 답변을 위해 최대 1~2분 소요될 수 있습니다.**  
        - **새로고침 시 채팅 내용이 사라지니 주의해주세요!**  
        """,
        icon="💡",
    )
    
    st.markdown("# **📝 예제 질문**")
    st.markdown(
        """
        - 🎯 **"코들 프로그램은 어떻게 사용하나요?"**  
        - 📌 **"출석 기준은 어떻게 되나요?"**  
        - 🔍 **"파이썬에서 리스트랑 튜플 차이가 뭐야?"**  
        
        💬 *답변이 모호하면 추가로 꼬리 질문을 할 수도 있어요!*
        """
    )

# 챗봇 상태 관리
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 대화 기록 저장
if "prev_question" not in st.session_state:
    st.session_state.prev_question = None
if "prev_ground_knowledge" not in st.session_state:
    st.session_state.prev_ground_knowledge = None
if "prev_answer" not in st.session_state:
    st.session_state.prev_answer = None
if "prev_question_type" not in st.session_state:
    st.session_state.prev_question_type = None
if "follow_up_question" not in st.session_state:
    st.session_state.follow_up_question = None
if "follow_up_clicked" not in st.session_state:
    st.session_state.follow_up_clicked = False

# AI 조교의 응답을 생성하는 함수
def chat_process(txt):
    key_knowledge = key_knowledge_generator.key_knowledge_generator(txt)
    question_type = ground_knowledge_generator.question_type_classifier(key_knowledge)
    ground_knowledge = ground_knowledge_generator.ground_knowledge_generator(key_knowledge)

    # "해당없음"일 경우 특정 메시지 반환
    if ground_knowledge.strip() == "해당없음":
        return ground_knowledge, "❌ 해당 질문은 답변할 수 없습니다. 담당 조교님 혹은 교수님께 직접 문의해보세요.", "해당없음", None
    elif ground_knowledge.strip() == "코딩":
        follow_up_question = follow_up_question_generator.follow_up_question_generator(txt)
        coding_functions = [coding_question_process_one, coding_question_process_two]
        
        # 랜덤으로 선택 후 실행, 오류 발생 시 다른 함수 실행
        random.shuffle(coding_functions)  # 랜덤으로 순서 섞기
        for func in coding_functions:
            try:
                answer = func(txt)
                return ground_knowledge, answer, "코딩", follow_up_question
            except Exception as e:
                return
        return "코딩", answer_generator.answer_generator(txt, ground_knowledge), "코딩", follow_up_question
    else:
        answer = answer_generator.answer_generator(txt, ground_knowledge)
        return ground_knowledge, answer, question_type, None
    
# 채팅 UI 출력
for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "user" else "assistant"):
        st.markdown(message)

if st.session_state.follow_up_clicked and st.session_state.follow_up_question:
    # 후속질문을 처리
    st.session_state.follow_up_clicked = False  # 클릭 상태 해제
    
    # user 발화
    fup = st.session_state.follow_up_question
    st.session_state.chat_history.append(("user", fup))
    with st.chat_message("user"):
        st.markdown(fup)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("AI 조교가 생각 중입니다... 🤔")

    # 후속질문에 대한 답변 생성
    ground_knowledge, assistant_response, question_type, new_followup = chat_process(fup)

    placeholder.markdown(assistant_response)
    st.session_state.chat_history.append(("assistant", assistant_response))

    # DB 저장
    save_chat_to_db(fup, assistant_response, True, question_type)

    # 세션 갱신
    st.session_state.prev_question = fup
    st.session_state.prev_ground_knowledge = ground_knowledge
    st.session_state.prev_answer = assistant_response
    st.session_state.prev_question_type = question_type

    # 새 후속질문
    st.session_state.follow_up_question = new_followup

# ----------------------------------------------------
# 2) 새 user_input 처리
# ----------------------------------------------------
user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    # user 발화
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("AI 조교가 생각 중입니다... 🤔")

    # 꼬리 질문인지 판단
    if st.session_state.prev_question:
        is_tail_question = tail_question_decider.tail_question_generator(
            st.session_state.prev_question,
            user_input
        )
        if is_tail_question.lower() == "yes":
            # 꼬리 질문 처리
            is_main_question = False
            assistant_response = tail_question_process.tail_question_process(
                st.session_state.prev_question,
                st.session_state.prev_ground_knowledge,
                st.session_state.prev_answer,
                user_input
            )
            ground_knowledge = st.session_state.prev_ground_knowledge
            question_type = st.session_state.prev_question_type
            follow_up_question = None
        else:
            # 새 질문 처리
            is_main_question = True
            ground_knowledge, assistant_response, question_type, follow_up_question = chat_process(user_input)
    else:
        is_main_question = True
        ground_knowledge, assistant_response, question_type, follow_up_question = chat_process(user_input)

    placeholder.markdown(assistant_response)
    st.session_state.chat_history.append(("assistant", assistant_response))

    # DB 저장
    save_chat_to_db(user_input, assistant_response, is_main_question, question_type)

    # 세션 갱신
    st.session_state.prev_question = user_input
    st.session_state.prev_ground_knowledge = ground_knowledge
    st.session_state.prev_answer = assistant_response
    st.session_state.prev_question_type = question_type

    # 후속질문
    if assistant_response.startswith("❌"):
        st.session_state.follow_up_question = None
    else:
        st.session_state.follow_up_question = follow_up_question

# ----------------------------------------------------
# 3) 현재 후속질문이 있으면, "버튼"을 즉시 표시
# ----------------------------------------------------
if st.session_state.follow_up_question:
    pressed = st.button(st.session_state.follow_up_question)
    if pressed:
        st.session_state.follow_up_clicked = True
        st.rerun()  # 버튼 누르는 순간 재실행 → (1) 로직으로 진입

# 



