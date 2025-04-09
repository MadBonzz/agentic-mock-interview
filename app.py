import streamlit as st
import os
import json
import re
import uuid
import tempfile
from llama_cpp import Llama
import pymupdf4llm
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI

# Initialize session state variables if they don't exist
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'question_history' not in st.session_state:
    st.session_state.question_history = []
if 'answer_history' not in st.session_state:
    st.session_state.answer_history = []
if 'evaluation_history' not in st.session_state:
    st.session_state.evaluation_history = []
if 'resume_processed' not in st.session_state:
    st.session_state.resume_processed = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'points' not in st.session_state:
    st.session_state.points = []

# Prompts (moved from separate file)
initial_prompt_template = """
You are an experienced technical interviewer. Based on the job description and resume information provided, ask a relevant first interview question to assess the candidate's fit for the role.

Job Description:
{job_description}

Resume Information:
{context}

Create a thoughtful first question that will help evaluate if the candidate is a good match for this position.
"""

prompt_template = """
You are an experienced technical interviewer evaluating a candidate's answer. Based on the job description, resume information, and the candidate's answer, provide a comprehensive evaluation.

Resume Information:
{context}

{answer}

Evaluate the technical accuracy, depth, and clarity of this answer. Then, create a follow-up question that will either:
1. Dig deeper if the answer was strong
2. Change to a new topic if appropriate
3. Ask for clarification if the answer wasn't sufficient

Your evaluation should be thorough and fair.
"""

# Main app layout
st.title("AI Mock Interview Assistant")
st.write("Upload your resume and provide a job description to start a personalized mock interview")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    server_url = st.text_input("LM Studio Server URL", value="http://localhost:1234/v1")
    llm_id = st.text_input("LLM Model ID", value="meta-llama-3.1-8b-instruct")
    
    # Path for embedding model
    embedding_model_path = st.text_input(
        "Path to Embedding Model", 
        value="C:/Users/shour/.cache/lm-studio/models/second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q4_0.gguf"
    )
    
    collection_name = st.text_input("Vector DB Collection Name", value="resume")
    
    # Advanced settings toggle
    show_advanced = st.checkbox("Show Advanced Settings")
    if show_advanced:
        st.subheader("Advanced Settings")
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        # Add other advanced settings here

# Upload resume
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

# Job description input
job_description = st.text_area("Enter the job description", height=300)

# Function to process the resume
def process_resume(pdf_file, embedding_model_path):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name
    
    # Process the PDF to markdown
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # Split text based on headers
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[('##', 'header1'), ('####', 'header2')]
        )
        texts = splitter.split_text(md_text)
        
        # Create embeddings
        embed_model = Llama(model_path=embedding_model_path, 
                          embedding=True,
                          verbose=False)
        
        chunks = []
        for i in range(len(texts)):
            chunk = dict()
            chunk['content'] = texts[i].page_content
            chunk['length'] = len(texts[i].page_content)
            chunk['embedding'] = embed_model.create_embedding(texts[i].page_content)['data'][0]['embedding']
            chunks.append(chunk)
        
        # Save chunks to session state
        st.session_state.chunks = chunks
        
        # Create vector DB points
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk['embedding'],
                payload={"text": chunk['content']}
            )
            for chunk in chunks
        ]
        st.session_state.points = points
        
        # Clean up temp file
        os.unlink(pdf_path)
        
        return True
    except Exception as e:
        st.error(f"Error processing resume: {str(e)}")
        # Clean up temp file
        try:
            os.unlink(pdf_path)
        except:
            pass
        return False

# Function to initialize the vector database
def setup_vector_db(collection_name, points):
    try:
        # Initialize Qdrant client
        client = QdrantClient(path="embeddings")
        
        # Check if collection exists and delete if it does
        try:
            if client.get_collection(collection_name):
                client.delete_collection(collection_name=collection_name)
        except ValueError:
            pass  # Collection doesn't exist
        
        # Create new collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        
        # Insert points
        operation_info = client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points
        )
        
        return client
    except Exception as e:
        st.error(f"Error setting up vector database: {str(e)}")
        return None

# Function to search the vector database
def search_vector_db(client, collection_name, query_vector, limit=3):
    try:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return search_result
    except Exception as e:
        st.error(f"Error searching vector database: {str(e)}")
        return []

# Function to get the first interview question
def get_first_question(gen_client, llm_id, context, job_description):
    try:
        first_response = gen_client.chat.completions.create(
            model=llm_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI interviewer following a structured process."
                },
                {
                    "role": "user",
                    "content": initial_prompt_template.format(
                        context=context,
                        job_description=job_description
                    )
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Information-Parser",
                        "parameters": {
                            "type": "object",
                            "title": "Information-Parser",
                            "properties": {
                                "question": {
                                    "title": "Question",
                                    "type": "string",
                                    "description": "The interview question being asked to the candidate."
                                },
                                "question_category": {
                                    "title": "Question Category",
                                    "type": "string",
                                    "description": "Category of the question, must be one of: Personality, Technical Concept, Experience.",
                                    "enum": ["Personality", "Technical Concept", "Experience"]
                                }
                            },
                            "required": [
                                "question",
                                "question_category"
                            ]
                        }
                    }
                }
            ],
            tool_choice={"type": "function", "function": {"name": "Information-Parser"}},
            temperature=1,
        )
        
        initial_question = eval(first_response.choices[0].message.tool_calls[0].function.arguments)
        return initial_question
    except Exception as e:
        st.error(f"Error getting first question: {str(e)}")
        return {"question": "Could not generate a question. Please try again.", "question_category": "Error"}

# Function to evaluate the answer and get the next question
def evaluate_answer(gen_client, llm_id, context, question, answer):
    try:
        response = gen_client.chat.completions.create(
            model=llm_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI interviewer following a structured process."
                },
                {
                    "role": "user",
                    "content": prompt_template.format(
                        context=context,
                        answer=f"Question : {question} Answer : {answer}"
                    )
                }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "InterviewEvaluation",
                        "parameters": {
                            "type": "object",
                            "title": "InterviewEvaluation",
                            "properties": {
                                "technical_ability": {
                                    "title": "Technical Ability",
                                    "type": "integer",
                                    "description": "Rating for technical correctness and depth of answer (0-10).",
                                    "minimum": 0,
                                    "maximum": 10
                                },
                                "language_clarity": {
                                    "title": "Language & Clarity",
                                    "type": "integer",
                                    "description": "Rating for communication, structure, and articulation (0-10).",
                                    "minimum": 0,
                                    "maximum": 10
                                },
                                "depth_of_knowledge": {
                                    "title": "Depth of Knowledge",
                                    "type": "integer",
                                    "description": "Rating for ability to explain concepts beyond surface level (0-10).",
                                    "minimum": 0,
                                    "maximum": 10
                                },
                                "review": {
                                    "title": "Review",
                                    "type": "string",
                                    "description": "Detailed feedback including strengths and areas for improvement."
                                },
                                "satisfaction_level": {
                                    "title": "Satisfaction Level",
                                    "type": "string",
                                    "description": "Overall assessment of the candidate's answer.",
                                    "enum": ["Highly Satisfactory", "Satisfactory", "Needs Improvement", "Unsatisfactory"]
                                },
                                "next_step": {
                                    "title": "Next Step",
                                    "type": "string",
                                    "description": "Determines the next action to take based on the answer.",
                                    "enum": ["Ask a deeper question", "Move to a new topic", "Prompt for a better answer"]
                                },
                                "question": {
                                    "title": "Question",
                                    "type": "string",
                                    "description": "The interview question being asked to the candidate. If the next step is to switch topics, ask about something else. In other scenarios, ask a more detailed or complex question."
                                },
                                "question_category": {
                                    "title": "Question Category",
                                    "type": "string",
                                    "description": "Category of the question, must be one of: Personality, Technical Concept, Experience.",
                                    "enum": ["Personality", "Technical Concept", "Experience"]
                                }
                            },
                            "required": [
                                "technical_ability",
                                "language_clarity",
                                "depth_of_knowledge",
                                "review",
                                "satisfaction_level",
                                "next_step",
                                "question",
                                "question_category"
                            ]
                        }
                    }
                }
            ],
            tool_choice={"type": "function", "function": {"name": "InterviewEvaluation"}},
            temperature=1,
        )
        
        parsed_response = response.choices[0].message.content
        match = re.search(r'"parameters": (\{.*?\})', parsed_response)
        if match:
            parameters_json = match.group(1)
            parameters_dict = json.loads(parameters_json)
            return parameters_dict
        else:
            match = re.search(r'tool_calls.*?function.*?arguments.*?(\{.*\})', parsed_response, re.DOTALL)
            if match:
                parameters_json = match.group(1)
                parameters_dict = json.loads(parameters_json)
                return parameters_dict
            else:
                st.error("Could not parse the response from the AI interviewer.")
                return None
    except Exception as e:
        st.error(f"Error evaluating answer: {str(e)}")
        return None

# Process button for resume and job description
if uploaded_file and job_description and st.button("Process Resume & Prepare Interview"):
    with st.spinner("Processing resume and setting up the interview..."):
        # Process the resume
        if process_resume(uploaded_file, embedding_model_path):
            st.session_state.resume_processed = True
            st.success("Resume processed successfully!")
        else:
            st.error("Failed to process resume.")

# Start Interview button
if st.session_state.resume_processed and st.button("Start Interview"):
    with st.spinner("Starting the interview..."):
        try:
            # Setup vector DB
            client = setup_vector_db(collection_name, st.session_state.points)
            
            if client:
                # Setup OpenAI client
                gen_client = OpenAI(base_url=server_url, api_key="lm-studio")
                
                # Get embedding for job description
                embed_model = Llama(model_path=embedding_model_path, 
                                  embedding=True,
                                  verbose=False)
                query_vector = embed_model.create_embedding(job_description)['data'][0]['embedding']
                
                # Search the vector DB
                search_result = search_vector_db(client, collection_name, query_vector)
                
                if search_result:
                    context = "\n\n".join([row.payload['text'] for row in search_result])
                    
                    # Get the first question
                    initial_question_data = get_first_question(gen_client, llm_id, context, job_description)
                    
                    if initial_question_data:
                        st.session_state.current_question = initial_question_data['question']
                        st.session_state.question_history.append({
                            "question": initial_question_data['question'],
                            "category": initial_question_data['question_category']
                        })
                        st.session_state.interview_started = True
                        st.success("Interview started successfully!")
                    else:
                        st.error("Failed to generate the first question.")
                else:
                    st.error("No relevant information found in the resume.")
        except Exception as e:
            st.error(f"Error starting the interview: {str(e)}")

# Display the current interview state
if st.session_state.interview_started:
    # Display current question
    st.subheader("Current Question:")
    st.write(st.session_state.current_question)
    
    # Input for answer
    answer = st.text_area("Your Answer:", height=150)
    
    # Submit answer
    if st.button("Submit Answer"):
        if answer:
            with st.spinner("Evaluating your answer..."):
                try:
                    # Setup OpenAI client
                    gen_client = OpenAI(base_url=server_url, api_key="lm-studio")
                    
                    # Get embedding for job description
                    embed_model = Llama(model_path=embedding_model_path, 
                                      embedding=True,
                                      verbose=False)
                    query_vector = embed_model.create_embedding(job_description)['data'][0]['embedding']
                    
                    # Setup vector DB client
                    client = QdrantClient(path="embeddings")
                    
                    # Search the vector DB
                    search_result = search_vector_db(client, collection_name, query_vector)
                    
                    if search_result:
                        context = "\n\n".join([row.payload['text'] for row in search_result])
                        
                        # Save the answer
                        st.session_state.answer_history.append(answer)
                        
                        # Evaluate the answer
                        evaluation = evaluate_answer(
                            gen_client, 
                            llm_id, 
                            context, 
                            st.session_state.current_question, 
                            answer
                        )
                        
                        if evaluation:
                            # Save the evaluation
                            st.session_state.evaluation_history.append(evaluation)
                            
                            # Update the current question
                            st.session_state.current_question = evaluation['question']
                            
                            # Add to question history
                            st.session_state.question_history.append({
                                "question": evaluation['question'],
                                "category": evaluation['question_category']
                            })
                            
                            # Show the evaluation
                            st.success("Answer submitted and evaluated!")
                            
                            # Rerun to update the UI
                            st.rerun()
                        else:
                            st.error("Failed to evaluate the answer.")
                    else:
                        st.error("No relevant information found in the resume.")
                except Exception as e:
                    st.error(f"Error evaluating the answer: {str(e)}")
        else:
            st.warning("Please provide an answer before submitting.")

    # Interview history and evaluations
    if st.session_state.evaluation_history:
        st.subheader("Interview Progress")
        
        with st.expander("View Interview History", expanded=False):
            for i, (question_data, answer, evaluation) in enumerate(zip(
                st.session_state.question_history[:-1], 
                st.session_state.answer_history, 
                st.session_state.evaluation_history
            )):
                st.markdown(f"**Question {i+1} ({question_data['category']}):** {question_data['question']}")
                st.markdown(f"**Your Answer:** {answer}")
                
                # Create metrics for the ratings
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Technical Ability", f"{evaluation['technical_ability']}/10")
                with col2:
                    st.metric("Language & Clarity", f"{evaluation['language_clarity']}/10")
                with col3:
                    st.metric("Depth of Knowledge", f"{evaluation['depth_of_knowledge']}/10")
                
                st.markdown(f"**Overall:** {evaluation['satisfaction_level']}")
                st.markdown(f"**Feedback:** {evaluation['review']}")
                st.markdown(f"**Next Step:** {evaluation['next_step']}")
                st.markdown("---")
        
        # Overall performance summary
        if len(st.session_state.evaluation_history) >= 3:
            with st.expander("Performance Summary", expanded=False):
                # Calculate averages
                avg_technical = sum(e['technical_ability'] for e in st.session_state.evaluation_history) / len(st.session_state.evaluation_history)
                avg_clarity = sum(e['language_clarity'] for e in st.session_state.evaluation_history) / len(st.session_state.evaluation_history)
                avg_depth = sum(e['depth_of_knowledge'] for e in st.session_state.evaluation_history) / len(st.session_state.evaluation_history)
                overall_score = (avg_technical + avg_clarity + avg_depth) / 3
                
                # Display summaries
                st.subheader("Overall Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg. Technical", f"{avg_technical:.1f}/10")
                with col2:
                    st.metric("Avg. Clarity", f"{avg_clarity:.1f}/10")
                with col3:
                    st.metric("Avg. Depth", f"{avg_depth:.1f}/10")
                with col4:
                    st.metric("Overall Score", f"{overall_score:.1f}/10")
                
                # Count satisfaction levels
                satisfaction_counts = {}
                for e in st.session_state.evaluation_history:
                    level = e['satisfaction_level']
                    satisfaction_counts[level] = satisfaction_counts.get(level, 0) + 1
                
                # Display satisfaction distribution
                st.subheader("Satisfaction Levels")
                for level, count in satisfaction_counts.items():
                    st.write(f"{level}: {count} question(s)")

if st.session_state.interview_started and st.button("End Interview"):
    if len(st.session_state.evaluation_history) > 0:
        st.session_state.interview_started = False
        
        # Calculate final score
        avg_technical = sum(e['technical_ability'] for e in st.session_state.evaluation_history) / len(st.session_state.evaluation_history)
        avg_clarity = sum(e['language_clarity'] for e in st.session_state.evaluation_history) / len(st.session_state.evaluation_history)
        avg_depth = sum(e['depth_of_knowledge'] for e in st.session_state.evaluation_history) / len(st.session_state.evaluation_history)
        overall_score = (avg_technical + avg_clarity + avg_depth) / 3
        
        # Display final score
        st.success("Interview completed!")
        st.subheader("Final Score")
        st.markdown(f"**Overall Score:** {overall_score:.1f}/10")
        
        # Generate final recommendations
        st.subheader("Recommendations")
        strengths = []
        improvements = []
        
        for evaluation in st.session_state.evaluation_history:
            # Extract strengths and areas for improvement from reviews
            review = evaluation['review']
            if "strength" in review.lower() or "good" in review.lower() or "excellent" in review.lower():
                strengths.append(review)
            if "improve" in review.lower() or "could" in review.lower() or "should" in review.lower():
                improvements.append(review)
        
        if strengths:
            st.markdown("**Strengths:**")
            for strength in strengths[:3]:  # Limit to 3 strengths
                st.markdown(f"- {strength}")
        
        if improvements:
            st.markdown("**Areas for Improvement:**")
            for improvement in improvements[:3]:  # Limit to 3 improvements
                st.markdown(f"- {improvement}")
        
        # Reset button
        if st.button("Start New Interview"):
            # Reset all session state variables
            st.session_state.interview_started = False
            st.session_state.current_question = ""
            st.session_state.question_history = []
            st.session_state.answer_history = []
            st.session_state.evaluation_history = []
            st.session_state.resume_processed = False
            st.session_state.chunks = []
            st.session_state.points = []
            st.rerun()
    else:
        st.warning("You haven't answered any questions yet. Please answer at least one question before ending the interview.")

with st.expander("How to Use This App", expanded=False):
    st.markdown("""
    ### How to Use This Mock Interview App
    
    1. **Upload your resume**: Start by uploading your resume in PDF format.
    2. **Enter the job description**: Paste the full job description for the position you're preparing for.
    3. **Process the resume**: Click 'Process Resume & Prepare Interview' to analyze your resume.
    4. **Start the interview**: Click 'Start Interview' to begin the mock interview process.
    5. **Answer questions**: Type your answers in the text area and click 'Submit Answer'.
    6. **Review feedback**: After each answer, you'll receive detailed feedback and a follow-up question.
    7. **View your progress**: Check the 'Interview Progress' section to see how you're doing.
    8. **End the interview**: When you're done, click 'End Interview' to see your final score and recommendations.
    
    ### Tips for a Successful Mock Interview
    
    - Be as detailed and specific as possible in your answers
    - Take your time to formulate thoughtful responses
    - Focus on highlighting relevant skills and experiences
    - Use concrete examples from your past work
    - Practice multiple times with different job descriptions
    """)

with st.expander("About", expanded=False):
    st.markdown("""
    **AI Mock Interview Assistant**
    
    This application uses local language models to create a personalized mock interview experience based on your resume and a target job description. It provides real-time feedback and evaluations to help you prepare for real interviews.
    
    The app leverages:
    - Local LLM models through LM Studio
    - Vector embeddings for resume analysis
    - Qdrant vector database for semantic search
    - Structured evaluation to provide meaningful feedback
    
    Note: All processing is done locally on your machine for privacy.
    """)