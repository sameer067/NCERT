import re
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def load_vectorstore(persist_directory="db"):
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def generate_quiz(topic: str, num_questions: int = 5):
    """Generate a quiz with MCQs from a topic in the NCERT book."""
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(topic)

    # Combine context into one string
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template("""
You are an educational quiz generator. Using only the information below from an NCERT textbook, create {num_questions} multiple choice questions on the topic: "{topic}".

Include 4 options per question (a/b/c/d) and clearly mark the correct answer. Format should be:

Q1: ...
a)
b)
c)
d)
Answer: b)

Context:
{context}
""")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    chain = prompt | llm

    response = chain.invoke({"topic": topic, "context": context, "num_questions": num_questions})
    return response.content

def run_interactive_quiz(topic):
    raw_quiz = generate_quiz(topic)
    
    # Split questions using regex
    questions = re.split(r'\n(?=Q\d+:)', raw_quiz)
    
    score = 0
    total = len(questions)

    print("\n Starting Quiz on:", topic)
    print("-" * 40)

    for q in questions:
        # Extract answer
        match = re.search(r'Answer:\s*([a-dA-D])\)?', q)
        if not match:
            continue  # skip malformed questions
        correct = match.group(1).lower()

        # Show question without the answer
        question_text = re.sub(r'Answer:.*', '', q).strip()
        print(question_text)

        # Ask user
        user_answer = input("Your answer (a/b/c/d): ").strip().lower()

        # Check and score
        if user_answer == correct:
            print("Correct!\n")
            score += 1
        else:
            print(f"Incorrect. Correct answer was: {correct}\n")

    print("=" * 40)
    print(f"Quiz complete! Your score: {score}/{total}")
    print("=" * 40)