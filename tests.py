import requests

response = requests.post(
      "https://llm-production-d199.up.railway.app/api/generate",
      json={
  "model": "qwen-career",
  "messages": [
    {
      "role": "user",
      "content": "Hey how are you"
    }
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
  )
print(response.json())

