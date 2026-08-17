from openai import OpenAI
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    gender: str
    occupation: str

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)

response = client.chat.completions.parse(
    model="mlx-community/gemma-4-e2b-it-4bit",
    messages=[
        {"role": "user", "content": "Tell me about Donald Trump in the format that I provided."}
    ],
    response_format=Person
)

person = response.choices[0].message.parsed

print(person.name)        # John
print(person.age)         # 42
print(person.occupation)
print(person.gender)