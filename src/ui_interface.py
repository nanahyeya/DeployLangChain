# src/ui_interface.py
import gradio as gr
from gradio_pdf import PDF
from rag_service import PDFRAGService


class PDFChatInterface:
    """PDF 채팅 인터페이스 클래스"""
    
    def __init__(self):
        self.rag_service = PDFRAGService()
    
    def process_pdf_and_answer(self, message, history, pdf_file, chunk_size, chunk_overlap, temperature):
        """PDF 처리 및 답변 생성"""
        if not pdf_file:
            return "PDF 파일을 업로드해주세요."
        
        if not message.strip():
            return "질문을 입력해주세요."
        
        try:
            # 새로운 PDF인지 확인하고 처리
            if not self.rag_service.is_pdf_loaded() or self.rag_service.current_pdf_path != pdf_file:
                print("새로운 PDF 파일 처리 중...")
                self.rag_service.load_pdf_to_vectorstore(pdf_file, chunk_size, chunk_overlap)
                print("PDF 처리 완료!")
            else:
                print("기존 벡터 저장소 사용")

            # 답변 생성
            answer = self.rag_service.get_answer(message, temperature)
            return answer
            
        except Exception as e:
            error_msg = f"처리 중 오류가 발생했습니다: {str(e)}"
            print(error_msg)
            return error_msg

    def respond(self, message, chat_history, pdf_file, chunk_size, chunk_overlap, temperature):
        """채팅 응답 처리"""
        if not message.strip():
            return chat_history, ""
        
        bot_message = self.process_pdf_and_answer(
            message, chat_history, pdf_file, chunk_size, chunk_overlap, temperature
        )
        
        chat_history.append((message, bot_message))
        return chat_history, ""

    def clear_chat(self):
        """채팅 기록 초기화"""
        return [], ""

    def set_example_question(self, question):
        """예시 질문 설정"""
        return question

    def create_interface(self):
        """Gradio 인터페이스 생성"""
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
                "이 문서에서 가장 중요한 핵심 사항은 무엇인가요?",
                "문서에 포함된 주요 절차나 단계를 알려주세요."
            ]
            
            example_buttons = []
            with gr.Row():
                for question in example_questions:
                    btn = gr.Button(question, size="sm")
                    example_buttons.append(btn)
            
            # 이벤트 핸들러 설정
            submit_btn.click(
                self.respond, 
                [msg, chatbot, pdf_input, chunk_size, chunk_overlap, temperature], 
                [chatbot, msg]
            )
            
            msg.submit(
                self.respond, 
                [msg, chatbot, pdf_input, chunk_size, chunk_overlap, temperature], 
                [chatbot, msg]
            )
            
            clear_btn.click(self.clear_chat, outputs=[chatbot, msg])
            
            # 예시 질문 버튼 이벤트
            for i, btn in enumerate(example_buttons):
                btn.click(
                    lambda q=example_questions[i]: self.set_example_question(q),
                    outputs=msg
                )
        
        return demo