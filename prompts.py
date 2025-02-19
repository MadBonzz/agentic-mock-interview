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

{context}

Answer : {answer}

Once you believe all the relevant topics have been covered, end the interview.
"""