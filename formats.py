initial_prompt_template = """
Follow the steps as followed to generate the desired output. Make sure to generate only questions and not the answer. 
Generate only 1 question at a time. Do not generate multiple questions or answer under any scenario.
Step 1: Understanding the Context

Analyze the resume and job description to identify key skills, experiences, and job-specific requirements.
Select a relevant topic and categorize the question (Personality, Technical Concepts, Experience).
Step 2: Question Generation

Generate a single, specific question relevant to the candidate’s background and job role.
Ensure clarity and avoid vague or generic questions.
Output Format:
Question: [The actual question]
Category: [Personality / Technical Concepts / Experience]


Analyze resume & job description → Select topic
Generate a single relevant question → Categorize it

Resume Information : {context}

Job Description : {job_description}

Based on the resume information and the job description create an initial question to start the interview.
"""

prompt_template = """
Follow the steps as followed to generate the desired output. Make sure to generate only questions and not the answer. 
Generate only 1 question at a time. Do not generate multiple questions or answer under any scenario.

Step 1: Answer Evaluation

Assess the response based on:
Technical Ability (0-10): Accuracy and correctness.
Language & Clarity (0-10): Communication and structure.
Depth of Knowledge (0-10): Problem-solving and real-world application.
Identify strengths, areas for improvement, and satisfaction level.
Output Format:
Evaluation:
Technical Ability: X/10
Language & Clarity: X/10
Depth of Knowledge: X/10
Review: Strengths | Areas for Improvement
Satisfaction Level: Highly Satisfactory / Satisfactory / Needs Improvement / Unsatisfactory
Next Action: Ask a deeper question / Move to a new topic / Prompt for a better answer

Step 2: Follow-up Strategy

If Highly Satisfactory → Move to a new Topic
If Satisfactory → Move to a new topic or ask for more detail.
If Needs Improvement → Prompt for a better answer.
If Unsatisfactory → Switch topics.
Step 3: Ensuring Topic Coverage

Cover different skills, experiences, and projects.
Avoid getting stuck on one topic or framework.

Step 4: Handling Abusive Behavior

If inappropriate behavior is detected, terminate the interview immediately.
Final Workflow Summary

Analyze resume & job description → Select topic
Generate a single relevant question → Categorize it
Evaluate the response (Technical Ability, Language, Depth of Knowledge)
Determine next action based on satisfaction level
Ensure diverse topic coverage
Provide final assessment & overall rating
Handle abusive behavior (if any)

Step 5: Question Generation

Generate a single, specific question relevant to the candidate’s background and job role.
Ensure clarity and avoid vague or generic questions.
Output Format:
Question: [The actual question]
Category: [Personality / Technical Concepts / Experience]

Make sure to provide ratings, review and satisfaction level for each answer provided  by the user.
Make sure to generate only a single question at a time. I do not want multiple questions or sequence of conversations.
Wait for the user to respond to the question before generating another question and evaluation.

{context}

Answer : {answer}

Once you believe all the relevant topics have been covered, end the interview.
"""

initial_schema = {
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

general_schema = {
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
                                "description": "Overall assessment of the candidate’s answer.",
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
            
system_prompt = {
                "role": "system",
                "content": "You are an AI interviewer following a structured process."
            }