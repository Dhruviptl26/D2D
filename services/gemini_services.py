import google.generativeai as genai
from config.settings import GEMINI_MODEL

def generate_code_from_json(detections_json, uploaded_image_reference, selected_language):
    prompt = f"""
You are an expert UI-to-code generator.

I will provide:
1. A JSON structure describing UI components detected from a sketch.
2. An image reference of the UI (for rough visual understanding).
3. The target programming language or framework selected by the user.

Your task:
- Use the JSON data as the *primary source* for understanding the UI structure, hierarchy, and components.
- The image is only for visual context — do NOT replicate it exactly.
- Generate clean, well-structured, and production-ready code in the given language.
- Maintain a logical layout, proper spacing, and naming conventions for clarity.
- The output should be a working UI implementation matching the elements described in the JSON.
- Do not include explanations or markdown — return only the final code.

Inputs:
🧩 **Detected JSON structure:** {detections_json}
🖼️ **Reference image (for context only):** {uploaded_image_reference}
💬 **Target language/framework:** {selected_language}

Now generate the full UI code using the JSON structure in the given language/framework.
"""

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text
