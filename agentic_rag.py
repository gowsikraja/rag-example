from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from src.data_loader import load_all_documents

from dotenv import load_dotenv
load_dotenv()


class AgentState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    needs_retrieval: bool


class AgenticRag():

    def __init__(self, llm_model: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=llm_model)

        # This will be used later for embedding generation
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001")

        print(f"[INFO] Gemini LLM & Embedding model initialized: {llm_model}")
        self.getRetriver()

    def getRetriver(self):
        documents = load_all_documents("data")
        # create vector store
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.retriever = vectorstore.as_retriever(k=3)


    def decide_retrieval(self, state: AgentState) -> AgentState:
        """
        Decide if we need to retrieve documents based on the question
        """
        question = state["question"]

        # Simple heuristic: if question contains certain keywords, retrieve
        retrieval_keywords = ["what", "how", "explain", "describe", "tell me"]
        needs_retrieval = any(keyword in question.lower()
                              for keyword in retrieval_keywords)

        return {**state, "needs_retrieval": needs_retrieval}

    def retrieve_documents(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant documents based on the question
        """
        question = state["question"]
        documents = self.retriever.invoke(question)

        return {**state, "documents": documents}

    def generate_answer(self, state: AgentState) -> AgentState:
        """
        Generate an answer using the retrieved documents or direct response
        """
        question = state["question"]
        documents = state.get("documents", [])

        if documents:
            # RAG approach: use documents as context
            context = "\n\n".join([doc.page_content for doc in documents])
            prompt = f"""Based on the following context, answer the question:

            Context:
            {context}

            Question: {question}

            Answer:"""
        else:
            # Direct response without retrieval
            prompt = f"Answer the following question: {question}"

        response = self.llm.invoke(prompt)
        answer = response.content

        return {**state, "answer": answer}  # type: ignore

    def should_retrieve(self, state: AgentState) -> str:
        """
        Determine the next step based on retrieval decision
        """
        if state["needs_retrieval"]:
            return "retrieve"
        else:
            return "generate"

    def build_graph(self):

        # Create the state graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("decide", self.decide_retrieval)
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("generate", self.generate_answer)

        # Set entry point
        workflow.set_entry_point("decide")

        # Add conditional edges
        workflow.add_conditional_edges(
            "decide",
            self.should_retrieve,
            {
                "retrieve": "retrieve",
                "generate": "generate"
            }
        )

        # Add edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # Compile the graph
        self.app = workflow.compile()
        print("[INFO] RAG StateGraph workflow compiled successfully.")

    def ask_question(self, question: str):
        """
        Helper function to ask a question and get an answer
        """

        initial_state: AgentState = {
            "question": question,
            "documents": [],
            "answer": "",
            "needs_retrieval": False
        }

        result = self.app.invoke(initial_state)
        return result


if __name__ == "__main__":
    rag_config = AgenticRag()
    rag_config.build_graph()

    question = "What is Pre Hospitalization?"
    result = rag_config.ask_question(question)
    print("Question:", question)
    print("Answer:", result["answer"])
