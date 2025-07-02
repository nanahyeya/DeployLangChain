# 환경 변수에서 API 키 가져오기
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API 키 검증
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# langchain 패키지
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

# RAG Chain 구현을 위한 패키지
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# gradio 인터페이스를 위한 패키지
from gradio_pdf import PDF

# 전역 변수로 벡터 저장소 관리 (성능 향상을 위해)
current_vectorstore = None
current_pdf_path = None

# pdf 파일을 읽어서 벡터 저장소에 저장
def load_pdf_to_vector_store(pdf_file, chunk_size=1000, chunk_overlap=100):
    try:
        print(f"PDF 파일 로딩 중: {pdf_file}")
        
        # PDF 파일 로딩
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        
        if not documents:
            raise ValueError("PDF 파일에서 텍스트를 추출할 수 없습니다.")
        
        print(f"총 {len(documents)}페이지 로드됨")

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chunk_size), 
            chunk_overlap=int(chunk_overlap),
            separators=["\n\n", "\n", ".", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        print(f"총 {len(splits)}개 청크로 분할됨")

        # 임베딩 모델 생성
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
        
        # FAISS 벡터 저장소 생성
        print("FAISS 벡터 저장소 생성 중...")
        vectorstore = FAISS.from_documents(
            documents=splits, 
            embedding=embeddings
        )
        
        print("벡터 저장소 생성 완료!")
        return vectorstore
        
    except Exception as e:
        print(f"PDF 로딩 중 오류 발생: {str(e)}")
        raise e

# 벡터 저장소에서 문서를 검색하고 답변을 생성
def retrieve_and_generate_answers(vectorstore, message, temperature=0):
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )

        template = '''다음 문맥을 바탕으로 질문에 정확하게 답변해주세요. 
문맥에서 관련 정보를 찾을 수 없다면, "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답변해주세요.

<문맥>
{context}
</문맥>

질문: {input}

답변:'''

        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(
            model='gpt-3.5-turbo', 
            temperature=float(temperature),
            api_key=OPENAI_API_KEY
        )

        document_chain = create_stuff_documents_chain(model, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        response = rag_chain.invoke({'input': message})

        return response['answer']
        
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

# Gradio 인터페이스에서 사용할 함수
def process_pdf_and_answer(message, history, pdf_file, chunk_size, chunk_overlap, temperature):
    global current_vectorstore, current_pdf_path
    
    if not pdf_file:
        return "PDF 파일을 업로드해주세요."
    
    if not message.strip():
        return "질문을 입력해주세요."
    
    try:
        if current_vectorstore is None or current_pdf_path != pdf_file:
            print("새로운 PDF 파일 처리 중...")
            current_vectorstore = load_pdf_to_vector_store(
                pdf_file, chunk_size, chunk_overlap
            )
            current_pdf_path = pdf_file
            print("PDF 처리 완료!")
        else:
            print("기존 벡터 저장소 사용")

        answer = retrieve_and_generate_answers(current_vectorstore, message, temperature)
        return answer
        
    except Exception as e:
        error_msg = f"처리 중 오류가 발생했습니다: {str(e)}"
        print(error_msg)
        return error_msg

# Gradio 인터페이스 생성
def create_interface():
    with gr.Blocks(title="PDF 질의응답 시스템") as demo:
        gr.Markdown("# PDF 질의응답 시스템")
        gr.Markdown("PDF 파일을 업로드하고 질문하면 AI가 문서 내용을 바탕으로 답변해드립니다.")
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = PDF(label="PDF 파일 업로드")
                
                with gr.Accordion("고급 설정", open=False):
                    chunk_size = gr.Number(
                        label="청크 크기", 
                        value=1000, 
                        info="텍스트를 나누는 단위 (500-2000 권장)"
                    )
                    chunk_overlap = gr.Number(
                        label="청크 중복", 
                        value=200, 
                        info="청크 간 중복되는 문자 수 (50-300 권장)"
                    )
                    temperature = gr.Slider(
                        label="창의성 수준", 
                        minimum=0, 
                        maximum=1, 
                        step=0.1, 
                        value=0.0,
                        info="0: 정확성 우선, 1: 창의성 우선"
                    )
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="💬 대화", height=500)
                msg = gr.Textbox(
                    label="질문 입력", 
                    placeholder="PDF 내용에 대해 질문해주세요...",
                    lines=2
                )
                
                with gr.Row():
                    submit_btn = gr.Button("📤 질문하기", variant="primary")
                    clear_btn = gr.Button("🗑️ 대화 초기화")
        
        gr.Markdown("### 질문 예시")
        example_questions = [
            "문서의 주요 내용을 요약해주세요.",
            "기타소득에는 어떤 것들이 있나요?",
            "세율은 어떻게 적용되나요?"
        ]
        
        example_buttons = []
        with gr.Row():
            for question in example_questions:
                btn = gr.Button(question, size="sm")
                example_buttons.append(btn)
        
        def respond(message, chat_history, pdf_file, chunk_size, chunk_overlap, temperature):
            if not message.strip():
                return chat_history, ""
            
            bot_message = process_pdf_and_answer(
                message, chat_history, pdf_file, chunk_size, chunk_overlap, temperature
            )
            
            chat_history.append((message, bot_message))
            return chat_history, ""
        
        submit_btn.click(
            respond, 
            [msg, chatbot, pdf_input, chunk_size, chunk_overlap, temperature], 
            [chatbot, msg]
        )
        
        msg.submit(
            respond, 
            [msg, chatbot, pdf_input, chunk_size, chunk_overlap, temperature], 
            [chatbot, msg]
        )
        
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])
        
        for i, btn in enumerate(example_buttons):
            btn.click(
                lambda q=example_questions[i]: q,
                outputs=msg
            )
    
    return demo

# 인터페이스 실행
if __name__ == "__main__":
    # demo = create_interface()
    # demo.launch(
    #     share=False,
    #     debug=True,
    #     server_name="127.0.0.1",
    #     #server_name="0.0.0.0",  # 컨테이너 외부에서 접근 가능
    #     server_port=7860
    # )

    demo = create_interface()
    
    # Docker 환경 감지
    import socket
    try:
        # Docker 컨테이너 내부에서 실행 중인지 확인
        hostname = socket.gethostname()
        is_docker = os.path.exists('/.dockerenv') or hostname.startswith('pdf-rag-app')
    except:
        is_docker = False
    
    # Docker 환경에 따라 서버 설정 변경
    if is_docker:
        server_name = "0.0.0.0"  # Docker 컨테이너에서는 모든 인터페이스에서 접근 가능
        print(" Docker 환경에서 실행 중...")
    else:
        server_name = "127.0.0.1"  # 로컬 개발 환경
        print(" 로컬 환경에서 실행 중...")
    
    demo.launch(
        share=False,
        debug=True,
        server_name=server_name,
        server_port=7860
    )