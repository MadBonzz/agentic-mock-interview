from llama_cpp import Llama
import pymupdf4llm
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import time

pdf_path = "uploaded_file.pdf"

md_text = pymupdf4llm.to_markdown(pdf_path)

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[('##', 'header1'), ('####', 'header2')]
)

texts = splitter.split_text(md_text)

embed_model = Llama(model_path='C:/Users/shour/.cache/lm-studio/models/second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q4_0.gguf', 
                    embedding=True,
                    verbose=False)

chunks = []
for i in range(len(texts)):
    chunk = dict()
    chunk['content']   = texts[i].page_content
    chunk['length']    = len(texts[i].page_content)
    chunk['embedding'] = embed_model.create_embedding(texts[i].page_content)['data'][0]['embedding']
    chunks.append(chunk)

file_name = 'chunks.json'
with open(file_name, 'w') as file:
    json.dump(chunks, file, indent=4)

client = QdrantClient(path="embeddings")

try:
    if client.get_collection("resume"):
            client.delete_collection(collection_name="resume")
except ValueError:
    print("Collection not found")

client.create_collection(
    collection_name="resume",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

embeddings = [chunk['embedding'] for chunk in chunks]

points = [
  PointStruct(
    id = str(uuid.uuid4()),
    vector = chunk['embedding'],
    payload = {
      "text": chunk['content']
    }
  )
  for chunk in chunks]

operation_info = client.upsert(
    collection_name="resume",
    wait=True,
    points=points
)

search_query = """
Job Description:
We are seeking a skilled and experienced Angular/.NET/Ionic Developer to join our development team. In this role, you will be responsible for developing and maintaining web applications using Angular for the front-end, .NET for the back-end, and Ionic for mobile application development. You will collaborate with cross-functional teams to design, develop, and deploy high-quality software solutions.

Share your resume to inshita@intuitiveapps.com with these details-
• Experience in .NET-
• Experience in Angular-
• Experience in Ionic-
• Current CTC
• Expected CTC
• Notice period (Official/Negotiable up to)
• Last working Day
• Amount of offer (If holding any)
• Current Location
• Preferred Location

Responsibilities-
• Develop and maintain web applications using Angular, .NET, and Ionic frameworks.(Flutter is good to have)
• Collaborate with designers, back-end developers, and stakeholders to gather requirements and translate them into technical specifications.
• Write clean, scalable, and maintainable code.
• Implement responsive designs and ensure cross-browser compatibility.
• Integrate RESTful APIs and third-party services into the applications.
• Conduct thorough testing and debugging to ensure high-quality software delivery.
• Optimize applications for maximum speed and scalability.
• Stay up-to-date with emerging technologies and industry trends to continuously improve development processes and techniques.
• Participate in code reviews and provide constructive feedback to team members.
• Assist in troubleshooting and resolving issues reported by end-users.

Requirements-
• Proven experience in developing web applications using Angular, .NET, and Ionic frameworks.
• Strong understanding of front-end technologies such as HTML5, CSS3, and JavaScript.
• Proficient in C# and .NET framework for back-end development.
• Familiarity with RESTful APIs and JSON.
• Experience with version control systems, such as Git.
• Knowledge of responsive design principles and mobile-first development.
• Understanding of software development best practices, including code optimization, debugging, and testing.
• Ability to work independently and collaboratively in a team environment.
• Excellent problem-solving and communication skills.

Skills: angular,.net,ionic framework,flutter,mobile app development
"""

query_vector = embed_model.create_embedding(search_query)['data'][0]['embedding']
search_result = client.search(
  collection_name="resume",
  query_vector=query_vector,
  limit=3
)

template = """
You are an AI interviewer that evaluates candidates based on their resumes and the job description. Your goal is to conduct a structured technical interview, assessing the candidate’s skills, experience, and domain knowledge.
Your job is only to ask questions and evaluate answers. Do not create answers by yourself. Only ask 1 question at a time. It is very important that you only create 1 question, wait for the user to respond with an answer and then evaluate the anser.
Interview Process:
1. Question Generation:
Analyze the resume and job description.
Generate an initial relevant and specific question based on the candidate’s experience, skills, or job role.
Do not explain why the question was chosen.
Label the question under one of the following categories:
Personality
Technical Concepts
Experience
Only ask one question at a time.
2. Answer Evaluation:
Assess the candidate’s response based on the following parameters:
Technical Ability (0-10): Accuracy, correctness, and depth of knowledge.
Language & Clarity (0-10): Communication skills, articulation, and structure of the response.
Depth of Knowledge (0-10): Ability to go beyond surface-level explanations, problem-solving approach, and real-world application.
Provide a detailed review based on these scores, highlighting strengths and areas for improvement.
Mandatory: Always assign a score in each category for every answer.
3. Follow-up Strategy:
If the answer is satisfactory, either:
Ask a deeper or more specific question on the same topic.
Move to a new relevant topic from the resume or job description.
If the answer is incomplete or incorrect, prompt the candidate to clarify or improve their response.
If the candidate is unable to answer, assign a low rating and shift to another topic.
4. Completion & Final Evaluation:
Ensure the interview covers diverse topics and does not get stuck on a single framework/concept.
Cover all key skills, projects, and experiences from the resume and job description.
Once satisfied, provide a comprehensive final assessment, summarizing:
Strengths
Weaknesses
Overall rating
5. Handling Abusive Behavior:
If any abusive language or inappropriate behavior is detected, terminate the interview immediately without further questions.
Finally I want the response in the format : 
Question : The actual question
Category : Category of the question generated

Once the candidate answers, provide a detailed evaluation in the following format:
Evaluation:  

**Technical Ability:** X/10  
**Language & Clarity:** X/10  
**Depth of Knowledge:** X/10  

**Review:**  
- Strengths:  
- Areas for Improvement:  

**Satisfaction Level:**  
- Highly Satisfactory / Satisfactory / Needs Improvement / Unsatisfactory  

Next Action:  
- [Move to a new topic] / [Ask a deeper question] / [Ask a simpler question]  
Ensure every answer is rated, reviewed, and categorized properly.

Make sure to finally provide the quesiton. Do not forget to give the question. If the user gets stuck at a topic or the is unable to answer, switch to another topic.
Follow all the instructions properly. Do not forget to provide ratings for all the parameters mentioned above. Decided whether to switch topics, ask detailed question and ask simpler question based on the satisfaction rating of the answer.
Provide ratings and review for each question as it is very important for the intervieweee. Also provide a final review at the end of the interview. Go through these instructions again and again during the generation process to decide what to do.

{context}

Question: {question}
"""

llm = Llama(
  model_path="C:/Users/shour/.cache/lm-studio/models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
  n_ctx=16384,
  n_gpu_layers=-1,
  offload_kqv=True,
  n_threads=16
)

answer = ""
while answer != "end":
    start_time = time.time()
    stream = llm.create_chat_completion(
    messages = [
      {"role": "user", "content": template.format(
        context = "\n\n".join([row.payload['text'] for row in search_result]),
        question = search_query      
      )}
    ],
    )

    response = stream['choices'][0]['message']['content']
    end_time = time.time()
    print(response)
    print(f"Time taken to generate question : {end_time - start_time}")
    answer = input("Enter your answer : ")

    search_query = f"{response}\n{answer}"