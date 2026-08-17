import ollama

response = ollama.chat(
    model='qwen2.5vl:3b',
    messages=[{
        'role': 'user',
        'content': 'Describe what this image shows in 2-3 simple sentences suitable for a student.',
        'images': ['test_output.png']
    }]
)
print(response['message']['content'])