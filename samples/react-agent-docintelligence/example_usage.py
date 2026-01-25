"""Test Azure OpenAI directly using langchain-openai."""
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

def main():
    print("🚀 Starting Azure OpenAI Direct Test...")
    print(f"Azure OpenAI Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"Deployment Name: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
    print(f"API Version: {os.getenv('OPENAI_API_VERSION')}")
    print("="*80)
    
    try:
        # Create the chat model using direct Azure OpenAI
        # Note: o4-mini only supports temperature=1 (default)
        model = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=1.0,  # o4-mini only supports temperature=1
        )
        
        # Test message
        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content="Hello! Please introduce yourself in one sentence.")
        ]
        
        print("\n📨 Sending message...")
        print("-"*80)
        
        # Invoke the model
        response = model.invoke(messages)
        
        print("\n✅ Response:")
        print("="*80)
        print(response.content)
        print("="*80)
        print("\n✅ Test completed successfully!")
        print(f"\n📊 Token Usage: {response.response_metadata.get('token_usage', 'N/A')}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
