from ollama import Client

client = Client()

response = client.chat(
    model="llama3.1:8b",
    messages=[
        {"role": "user", "content": "Analyse ce texte : Ingénieure systèmes embarqués."}
    ]
)

print(response["message"]["content"])
