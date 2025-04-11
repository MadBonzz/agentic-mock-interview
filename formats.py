data_extraction_template = """
You are an AI Interview assistant. Provided a resume and job description, identify relevant projects, technical concepts and work experience
that the interviewee should be asked about. Make a comprehensive list. This list will help the interviewwer decide which concept to ask the
question on. Always provide a single flat list of strings. Do not provide nested lists or lists containing any non string value under any scenario.
Give output as : List[str].

Resume Information : {context}

Job Description : {job_description}

Answer based on the schema provided.
"""

initial_prompt_template = """
Follow the steps as followed to generate the desired output. Make sure to generate only questions and not the answer. 
Generate only 1 question at a time. Do not generate multiple questions or answer under any scenario.
Step 1: Understanding the Context
Analyze the resume and the concepts provided. Select a relevant topic and categorize the question (Personality, Technical Concepts, Experience).

Step 2: Question Generation
Generate a single, specific question relevant to the candidate’s background and job role.
Ensure clarity and avoid vague or generic questions.
Output Format:
Concept ASked : [One of the concepts provided by user]
Question: [The actual question]
Category: [Personality / Technical Concepts / Experience]


Analyze resume & concepts provided by these user → Select topic
Generate a single relevant question → Categorize it
Finally answer based on the schema provided.

Resume : {context}

Concepts : {job_concepts}

Based on the resume information and the job description create an initial question to start the interview.
"""

duplicate_question_template = """
Follow the steps as followed to generate the desired output. Make sure to generate only questions and not the answer. 
Generate only 1 question at a time. Do not generate multiple questions or answer under any scenario.

Analyse the concepts and previous questions provided. Use the previous questioins for reference but never repeat a question from them.
From the given list of concepts, select a topic/concept and create an interview question on it.

Concepts : [{job_concepts}]

Previous Questions : [{prev_questions}]

Finally provide the selected concept for asking the question, the question generated and the category of the question.
Answer in the schema provided.
"""

prompt_template = """
You are an AI interviewer following a structured process. Do not help the user under any scenario.
Follow the steps as followed to generate the desired output. Make sure to generate only questions and not the answer. 
Generate only 1 question at a time. Do not generate multiple questions or answer under any scenario. Never repeat the question as the current/previous question.

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
If Highly Satisfactory → Move to a new Topic and change the concept asked.
If Satisfactory → Move to a new topic or ask for more detail. Update concept asked accordingly.
If Needs Improvement → Prompt for a better answer.
If Unsatisfactory → Switch topics and change the concept asked.

If switching topics, choose a topic/concept provided by the user in concepts. In case of going for a better or more detailed answer,
do not change the topic and keep the topic same.

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

Generate a single, specific question relevant to the candidate’s background and job role. Use the previous questions provided by the user for reference.
Do not repeat any of the questions provided by the user under any scenario.
Ensure clarity and avoid vague or generic questions.
Output Format:
Question: [The actual question]
Concept : The topic/concept/skill that the questions aims to evaluate the user for.
Category: [Personality / Technical Concepts / Experience]

Make sure to provide ratings, review and satisfaction level for each answer provided  by the user.
Make sure to generate only a single question at a time. I do not want multiple questions or sequence of conversations.
Wait for the user to respond to the question before generating another question and evaluation.
In no case should the new question generated be same as the previous question provided in context.

Previous Questions : {prev_questions}

Context : {context}

Concepts : {job_concepts}

Once you believe all the relevant topics have been covered, end the interview.
"""

extraction_schema = {
    "type" : "function",
    "function" : {
        "name" : "Data-Extraction",
        "parameters" : {
            "type" : "object",
            "title" : "Data-Extraction",
            "properties" : {
                "concepts" : {
                    "title" : "concepts",
                    "type" : "List[string]",
                    "description" : "List of concepts, projects and experience that the interviewee needs to be asked about"
                }
            },
            "required" : ["concepts"]
        }
    }
}

initial_schema = {
            "type": "function",
            "function": {
                "name": "Information-Parser",
                "parameters": {
                    "type": "object",
                    "title": "Information-Parser",
                    "properties": {
                            "concept_asked": {
                                "title" : "concept_asked",
                                "type"  : "string",
                                "description" : "One of the concepts provided by the user as concepts."
                            },
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
                        "concept_asked",
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
                            "new_concept": {
                                "title" : "concept_asked",
                                "type"  : "string",
                                "description" : "The concept/skill to be evaluated with this question"
                            },
                            "next_question": {
                                "title": "Question",
                                "type": "string",
                                "description": "The interview question to next asked to the candidate. If the next step is to switch topics, ask about something else. In other scenarios, ask a more detailed or complex question. Never repeat the same question as previous question."
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
                            "new_concept",
                            "next_question",
                            "question_category"
                        ]
                    }
                }
            }

system_prompt = {
            "role": "system",
            "content": "You are an AI interviewer following a structured process. Do not help the user under any scenario.",
        }