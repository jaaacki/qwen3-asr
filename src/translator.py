import os
import time
from typing import Optional
from logger import log

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
        
    log.info("Translation request | model={} target={} text_len={}", model, lang_name, len(text))

    prompt = (
        f"Translate the following spoken audio transcription into {lang_name}. "
        f"Preserve the original meaning and tone. Output ONLY the translated text required "
        f"without any introduction, markdown blocks, quotes, or commentary.\n\nText: {text}"
    )

    t0 = time.time()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional and highly accurate translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
    except Exception as e:
        log.error("Translation API error | model={} target={} elapsed={:.2f}s error={}", model, lang_name, time.time() - t0, e)
        raise

    if not response.choices:
        raise ValueError("Translation returned no choices")

    result = response.choices[0].message.content.strip()
    log.info("Translation complete | model={} target={} in_len={} out_len={} elapsed={:.2f}s", model, lang_name, len(text), len(result), time.time() - t0)
    return result

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
        
    log.info("SRT translation request | model={} target={} srt_len={}", model, lang_name, len(srt_content))

    prompt = (
        f"Translate the following subtitle (SRT) content into {lang_name}. "
        f"Preserve the original SRT format and timing tags perfectly. "
        f"Output ONLY the valid translated SRT content without any introduction, markdown wrapping blocks (like ```srt), or commentary. "
        f"Do NOT change the SRT index numbers or timestamp lines.\n\nSRT Content:\n{srt_content}"
    )

    t0 = time.time()
    # Needs a slightly larger context/timeout for full SRT files, but for standard files this works
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional subtitle translator. You MUST output ONLY valid SRT format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
    except Exception as e:
        log.error("SRT translation API error | model={} target={} elapsed={:.2f}s error={}", model, lang_name, time.time() - t0, e)
        raise

    if not response.choices:
        raise ValueError("Translation returned no choices")

    result = response.choices[0].message.content.strip()

    # Strip markdown block if model ignored instructions
    if result.startswith("```"):
        log.debug("Stripped markdown wrapper from SRT translation output")
        lines = result.split("\n")
        if lines[0].startswith("```"):
            lines.pop(0)
        if lines[-1].startswith("```"):
            lines.pop(-1)
        result = "\n".join(lines).strip()

    log.info("SRT translation complete | model={} target={} in_len={} out_len={} elapsed={:.2f}s", model, lang_name, len(srt_content), len(result), time.time() - t0)
    return result
