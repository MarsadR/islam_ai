"""
Noor — Islamic AI Assistant API (FastAPI, for HF Spaces Docker SDK)
---------------------------------------------------------------------
A plain REST endpoint so this can be called directly from React Native
(or any HTTP client) without dealing with Gradio's queue/SSE API.

Local run:
    uvicorn app:app --host 0.0.0.0 --port 7860
    (needs a .env file with GROQ_API_KEY=your_key)

Endpoint (once deployed on HF Spaces, Docker SDK):
    POST https://<your-hf-username>-<space-name>.hf.space/chat
    Body: { "message": "string", "history": [{"role": "user", "content": "..."}] }
    Response: { "reply": "string" }
"""

import os
import time
from datetime import datetime, timezone, timedelta
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from supabase import create_client, Client

load_dotenv()  # loads .env locally; no-op on HF Spaces (uses Space Secrets instead)

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY not set. Locally: add it to .env. "
        "On HF Spaces: add it under Settings > Variables and secrets."
    )

client = Groq(api_key=API_KEY)
MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are "Ibrahim", an Islamic Knowledge Assistant. You can talk in multi-languages(English,urdu,arabic, hindi, punjabi, bengali, turkish, etc) and your responses should be in the same language as the user message.
Your sole purpose is to help users with questions related to Islam, and to
exchange greetings/small talk in a warm, respectful manner.

SCOPE — you MAY answer:
- The Quran (translation, tafsir, meaning, context of revelation)
- Hadith (meaning, context, authenticity/grading when known)
- Fiqh (jurisprudence) across the major madhabs (Hanafi, Shafi'i, Maliki, Hanbali)
- Aqeedah (creed/theology), Seerah (life of the Prophet), history of the Sahabah
- Islamic finance and ethics, worship practices (salah, sawm, zakat, hajj, etc.)
- Greetings such as "Assalamu Alaikum" and general friendly small talk

SCOPE — you must NOT answer:
- Anything unrelated to Islam (general trivia, coding help, entertainment,
  unrelated politics, etc.)
- If asked something out of scope, politely decline, e.g.:
  "I'm an Islamic knowledge assistant, so I can only help with questions
  related to Islam. Is there something about Islam I can help you with?"

ACCURACY RULES (CRITICAL — follow strictly):
1. NEVER fabricate or guess a Quran ayah number or Hadith reference. If you
   are not fully certain of the exact Surah:Ayah or Hadith collection/book/
   number, say so explicitly and tell the user to verify on a trusted source
   (e.g., quran.com, sunnah.com) rather than inventing a citation.
2. Always cite sources when making a religious claim:
   - Quran: "Surah [Name] [Chapter]:[Ayah]" e.g. "Surah Al-Baqarah 2:255"
   - Hadith: "[Collection], [Book/Chapter], Hadith [Number] ([Grade if known])"
     e.g. "Sahih al-Bukhari, Book of Faith, Hadith 8 (Sahih)"
3. When scholars or madhabs differ on a ruling, present the mainstream
   positions fairly instead of asserting a single view as the only truth.
4. For matters needing a personal fatwa (divorce, inheritance shares, complex
   contracts, medical/ethical dilemmas), give general guidance but clearly
   recommend consulting a qualified local scholar/mufti.
5. Never issue takfir (declaring someone a non-Muslim) and never make
   sectarian attacks. Be respectful of diversity within the Muslim ummah.
6. Use ﷺ after "Prophet Muhammad" and "(may Allah be pleased with him/her)"
   or "(RA)" for companions, where it reads naturally.
7. If uncertain about anything, say so plainly ("Allahu A'lam — Allah knows
   best") rather than answering confidently with unverified information.

RESPONSE FORMAT (for Islam-related questions, not for greetings/small talk):
1. Start with a clear, direct answer/explanation in plain language.
2. End with a "References:" section listing the Quran ayat and/or Hadith
   used to support the answer, so the user can verify if they wish.
   Example:

   [Direct answer explaining the ruling/concept in 2-5 sentences or bullets]

   References:
   - Surah Al-Baqarah 2:183
   - Sahih al-Bukhari, Book of Fasting, Hadith 1904 (Sahih)

3. If you are not certain of an exact reference, still give the general
   answer, but say so plainly in the References section instead of
   inventing a citation.
4. For greetings and small talk, skip this structure and just respond
   naturally and warmly.

TONE: Warm, humble, respectful.
Keep answers clear and well-organized; use short paragraphs or bullet points
for multi-part answers.
"""


# Supabase integration for persistent token tracking
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    print("WARNING: SUPABASE_URL and SUPABASE_KEY not found. Token limiting will fallback to in-memory (resets on Vercel cold starts).")

# In-memory dictionary as fallback
user_token_usage = {}
TOKEN_LIMIT = 4500
RESET_TIME_SECONDS = 5 * 3600  # 5 hours


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[HistoryMessage]] = []
    user_id: Optional[str] = None  # To identify the user for token limiting


class ChatResponse(BaseModel):
    reply: str
    tokens_used: Optional[int] = None
    tokens_remaining: Optional[int] = None


app = FastAPI(title="Noor Islamic AI Assistant API")

# CORS open by default so the React Native app can call this freely.
# Tighten allow_origins if you later add a web frontend that needs restricting.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api")
def root():
    return {"status": "ok", "service": "Noor Islamic AI Assistant"}


@app.get("/api/health")
def health():
    return {"status": "healthy"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    user_id = request.user_id or "anonymous"
    current_tokens = 0

    if supabase is not None and user_id != "anonymous":
        try:
            # Check Supabase for user's token usage
            db_response = supabase.table("user_token_usage").select("*").eq("user_id", user_id).execute()
            if len(db_response.data) > 0:
                record = db_response.data[0]
                reset_time_str = record.get("reset_time")
                current_tokens = record.get("tokens_used", 0)
                
                # Check if 5 hours have passed
                reset_time = datetime.fromisoformat(reset_time_str.replace("Z", "+00:00"))
                if datetime.now(timezone.utc) > reset_time:
                    # Time has passed, reset tokens
                    current_tokens = 0
                    new_reset_time = datetime.now(timezone.utc) + timedelta(hours=5)
                    supabase.table("user_token_usage").update({
                        "tokens_used": 0,
                        "reset_time": new_reset_time.isoformat()
                    }).eq("user_id", user_id).execute()
            else:
                # Create record for new user
                new_reset_time = datetime.now(timezone.utc) + timedelta(hours=5)
                supabase.table("user_token_usage").insert({
                    "user_id": user_id,
                    "tokens_used": 0,
                    "reset_time": new_reset_time.isoformat()
                }).execute()
        except Exception as e:
            print(f"Supabase error checking tokens: {e}")
            # Fallback to allow request if DB fails to avoid blocking users
            pass
            
        if current_tokens >= TOKEN_LIMIT:
            raise HTTPException(
                status_code=429, 
                detail=f"Token limit exceeded. You can only use {TOKEN_LIMIT} tokens per 5 hours. Please try again later."
            )
    else:
        # Fallback in-memory logic
        current_time = time.time()
        if user_id not in user_token_usage:
            user_token_usage[user_id] = {"tokens": 0, "reset_time": current_time + RESET_TIME_SECONDS}
        else:
            if current_time > user_token_usage[user_id]["reset_time"]:
                user_token_usage[user_id] = {"tokens": 0, "reset_time": current_time + RESET_TIME_SECONDS}

        if user_token_usage[user_id]["tokens"] >= TOKEN_LIMIT:
            raise HTTPException(
                status_code=429, 
                detail=f"Token limit exceeded. You can only use {TOKEN_LIMIT} tokens per 5 hours. Please try again later."
            )
        current_tokens = user_token_usage[user_id]["tokens"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend([m.model_dump() for m in request.history])
    messages.append({"role": "user", "content": request.message})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=2500,
        )
        reply = response.choices[0].message.content
        
        # Calculate tokens used (Groq provides usage stats similar to OpenAI)
        tokens_used = response.usage.total_tokens if response.usage else 0
        
        # Update user's token usage
        if supabase is not None and user_id != "anonymous":
            try:
                supabase.table("user_token_usage").update({
                    "tokens_used": current_tokens + tokens_used
                }).eq("user_id", user_id).execute()
            except Exception as e:
                print(f"Supabase error updating tokens: {e}")
        else:
            user_token_usage[user_id]["tokens"] += tokens_used

        return ChatResponse(
            reply=reply,
            tokens_used=tokens_used,
            tokens_remaining=max(0, TOKEN_LIMIT - (current_tokens + tokens_used))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {e}")