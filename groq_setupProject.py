from groq import Groq                    # Groq client for model interactions
import httpx
import ssl
import AppConfig.Confighelper as ConHP 
import os

# Configure SSL settings to bypass certificate verification
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Create HTTP client with SSL verification disabled
http_client = httpx.Client(verify=False, 
                           follow_redirects=True,  # Handle 307 redirects
                            timeout=30.0            # Set timeout to avoid hanging
                            )

# GROQ API KEY
os.environ['GROQ_API_KEY'] = ConHP.get_config('Grok_API_KEY')

# Initialize client with your API key
groq_client = Groq(api_key=os.environ['GROQ_API_KEY'], http_client=http_client)

groq_model = "llama3-8b-8192"
# List all available models
models = groq_client.models.list()

print("Available Groq models:", "".join(str(model) for model in models))

def llm_call_groq(prompt, data=None, base64_img=None):
    """
    Makes a call to Groq to process text data.
    Note: Groq doesn't support vision/image analysis currently.
    
    Args:
        prompt (str): The instruction prompt to send to the LLM
        data (str, optional): Text data to be analyzed (e.g., extracted PDF text)
        base64_img (str, optional): Not supported by Groq - will be ignored
    
    Returns:
        str: The LLM's response content, or None if an error occurs
    """
    try:
        # Combine prompt with data if provided
        full_prompt = prompt
        if data:
            full_prompt = f"{prompt}\n\n{data}"
        
        # Warning if image is provided (Groq doesn't support vision)
        if base64_img:
            full_prompt = f"{prompt} \n\n Base64:\n{base64_img}"

        # Make the API call to Groq
        response = client.chat.completions.create(
            model=groq_model,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            max_tokens=4000,  # Adjust based on model limits
            temperature=0.1   # Lower temperature for more consistent outputs
        )

        # Extract and return the response content
        return response.choices[0].message.content

    except Exception as e:
        print(f"\nError during Groq LLM call: {e}\n")
        return None
