# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

# Load API keys
KEYS_FILE = "keys/deepseek.txt"

def load_key(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        exit(1)

# Load API key
api_key = load_key(KEYS_FILE)

# Initialize OpenAI client with DeepSeek base URL
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# DeepSeek Chat Function

def deepseek(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

# Test the function
if __name__ == "__main__":
    user_input = input("Enter your query for DeepSeek: ")
    print(deepseek(user_input))
