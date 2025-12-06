import google.generativeai as genai
import json

class quizGenerator:
    def __init__(self,api_key:str):
        """
        Docstring for __init__
        
        :param api_key: Description
        :type api_key: str
        """
        genai.configure(api_key=api_key)
        # Using Gemini 2.0 Flash (experimental) - fastest model for quiz generation
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')

    def generate_quiz(self,topic:str,topic_content:str)->json:
        
        prompt=f""" your are provided with a topic and content of the topic 
        topic/title:{topic}
        
        Instruction:
            - your work is to generate quiz from the content provided.
            - extract the key important concepts in the corpus and generate quiz on it.
            - write a clean moderate level quizes.
            - give questions along with their option,correct option , explanation.

        Return the results as a JSON array with this structure:
        [
            {{
                "Question": "give here the question",
                "options": ["option1,"option2","option3","option4"],
                "answer":"correct option to the question",
                "explanation":"give  two line explanation for the correct answer"
            }}
        ]
        
        Text segment:{topic_content}

        CRITICAL RULES:
            - write 5 quizes.
            - Return ONLY valid JSON, no additional text.
        """

        try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()

                # Remove markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]

                response_text = response_text.strip()

                questions = json.loads(response_text)
                
                print(f"Found {len(questions)} topics in chunk {questions}")
                
                return questions

        except json.JSONDecodeError as e:
            print(f"JSON parsing error in chunk {questions}: {e}")
            print(f"Response: {response_text[:200]}")
            
        except Exception as e:
            print(f"Error processing chunk {questions}: {e}")

    
            

