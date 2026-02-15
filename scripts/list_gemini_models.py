"""List available Gemini models for your API key."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY in .env")
        sys.exit(1)

    import google.generativeai as genai
    genai.configure(api_key=api_key)

    print("Available Gemini models:\n")
    for m in genai.list_models():
        if "generateContent" in (m.supported_generation_methods or []):
            print(f"  {m.name}")

if __name__ == "__main__":
    main()
