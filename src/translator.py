import os
from typing import Optional

def _get_client():
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("The 'openai' python package is required for translation. Please run `pip install openai`.")
    
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")  # Default to EMPTY for local APIs like Ollama
    base_url = os.getenv("OPENAI_BASE_URL")
    
    # If using Ollama, base_url is typically http://localhost:11434/v1
    
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
        
    return AsyncOpenAI(**client_kwargs)

async def translate_text(text: str, target_lang: str) -> str:
    """Translate raw text to the specified target language using an OpenAI-compatible API."""
    if not text.strip():
        return text
        
    client = _get_client()
    model = os.getenv("TRANSLATE_MODEL", "gpt-3.5-turbo")
    
    if target_lang.lower() == "en" or target_lang.lower() == "english":
        lang_name = "English"
    elif target_lang.lower() == "zh" or target_lang.lower() == "chinese":
        lang_name = "Chinese"
    else:
        lang_name = target_lang
        
    prompt = (
        f"Translate the following spoken audio transcription into {lang_name}. "
        f"Preserve the original meaning and tone. Output ONLY the translated text required "
        f"without any introduction, markdown blocks, quotes, or commentary.\n\nText: {text}"
    )
    
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional and highly accurate translator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    
    if not response.choices:
        raise ValueError("Translation returned no choices")
        
    return response.choices[0].message.content.strip()

async def translate_srt(srt_content: str, target_lang: str) -> str:
    """Translate SRT subtitle content to the specified target language, preserving timestamps."""
    if not srt_content.strip():
        return srt_content
        
    client = _get_client()
    model = os.getenv("TRANSLATE_MODEL", "gpt-3.5-turbo")
    
    if target_lang.lower() == "en" or target_lang.lower() == "english":
        lang_name = "English"
    elif target_lang.lower() == "zh" or target_lang.lower() == "chinese":
        lang_name = "Chinese"
    else:
        lang_name = target_lang
        
    prompt = (
        f"Translate the following subtitle (SRT) content into {lang_name}. "
        f"Preserve the original SRT format and timing tags perfectly. "
        f"Output ONLY the valid translated SRT content without any introduction, markdown wrapping blocks (like ```srt), or commentary. "
        f"Do NOT change the SRT index numbers or timestamp lines.\n\nSRT Content:\n{srt_content}"
    )
    
    # Needs a slightly larger context/timeout for full SRT files, but for standard files this works
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional subtitle translator. You MUST output ONLY valid SRT format."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )
    
    if not response.choices:
        raise ValueError("Translation returned no choices")
        
    result = response.choices[0].message.content.strip()
    
    # Strip markdown block if model ignored instructions
    if result.startswith("```"):
        lines = result.split("\n")
        if lines[0].startswith("```"):
            lines.pop(0)
        if lines[-1].startswith("```"):
            lines.pop(-1)
        result = "\n".join(lines).strip()
        
    return result
