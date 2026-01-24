# BrainBuddy-NexGen-Tutor
NexGen-Tutor
Brain Buddy Subject Tutor: AI-Powered Secondary Education 
Project Title: NexGen Tutor
Team Members: 
•	Syed Muhammad Rafay Shah
•	Hassan Asrar
•	Mahnoor Adil
Problem: A student thriving through immense depression before exams, Why?. Just because doubts weren’t clear and no real-time example exists? A number of students struggle through hundreds of pages and piles of books, yet are unbale to find the rightful information
Core Objective: An adaptive tutor for Grades 6–10, specializing in Science, History, Chemistry, Biology, Math and Geography with automated curriculum alignment,
Solution: We came up with something that doesn’t provide banal information  from any source unlike other LLMs like Gemini, ChatGPT, or any others and answers only specifically and strictly educational purpose. It helps the student clear modern world doubts with real time examples for a more clear overview.
______________
1. Tech Stack & Architecture
•	Environment: PyCharm (IDE) / Python 3.x
•	Frontend/UI: Streamlit (Custom-styled with Dark/Light cosmic and platinum silver themes). For Frontend Visual Representation.
•	RAG Framework: LangChain with ChromaDB for local vector storage and semantic retrieval.
•	Libraries used: We compiled all the libraries in a text file and then executed that text file through PyCharm Terminal by using the pip install command.                                                             The libraries included; 1-LangChains(for communication between our bot and any LLM), 2-Streamlit( for Visual display on website), 3- Edge-TTS(for conversion of text into Mp3 audio), etc.                                                           
•	Tools Used: We used multiple tools of different models to operate our program and install various features, like; 1-Google Generative AI, for embeddings and chatting with the LLM, 2-PYPDF loader, to load raw documents, 3-Text Splitters, to split documents for easy and accurate answers.
•	Styling of interface: We used CSS for styling the webpage.                            
•	Models: Gemini 2.5 Flash (LLM) & Google Generative AI Embeddings.
•	Audio Engine: Edge-TTS (Asynchronous processing for instant high-fidelity voice output).
______________2. Feature Highlights
•	Adaptive Complexity: Dynamic prompt engineering that scales language depth based on student grade level (6–10).
•	Hybrid Knowledge Base: Combines pre-authorized textbook libraries with a real-time PDF upload/sync feature.
•	Automated Quiz Me: Uses NLP to extract session topics and generate 3-question MCQ knowledge checks on demand.
•	Academic Guardrails: Hard-coded logic ensuring neutrality in history and refusal of non-educational content.
______________3. AI Usage Disclosure
•	Tools: Gemini 2.0 Flash, GitHub Copilot, and Claude 3.5.
•	Weight: ~35% of the project was AI-assisted.
•	Focus Areas:
o	Logic: Structuring asynchronous Python loops for audio/UI synchronization.
o	UI/UX: CSS-in-Markdown styling and theme transitions.
o	Refinement: Debugging session-state persistence and JSON-based chat history.
______________4. Resource & Compliance
•	Data Integrity: All pre-loaded materials are sourced from verified educational PDFs.
•	Open Library: Supports external syncing of additional teacher-provided resources to ensure the tutor remains current.
5. Detailed Solution
Our project uses RAG(for strictly educational purpose and relevant extraction of answers) for semantic search to be triggered as soon as the user demands an answer, and provides not all references but specific answers, also allowing users or students to sync their local libraries/external educational content or files. Provides a user friendly environment to give grade appropriate answers in accordance to the subject selected, and then also provides a 3 MCQs based quiz based on selection of a particular topic from the saved chats or a custom written topic, moreover corrects your doubts for the wrong answers selected. Lastly the most fun part, our PROGRESS DASHBOARD that provides a brief progress of the students understanding based on the quizzes attempted.
