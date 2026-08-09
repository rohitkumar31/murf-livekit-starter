import json
import logging
import sqlite3
from datetime import datetime, timezone

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    tokenize,
    room_io,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# ---------------------------------------------------------------------------
# Simple SQLite memory store for returning callers
# ---------------------------------------------------------------------------
DB_PATH = "saathi_memory.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS callers (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            language_preference TEXT,
            facts TEXT,
            last_interaction TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def get_caller(user_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT user_id, name, language_preference, facts, last_interaction FROM callers WHERE user_id = ?",
        (user_id.strip().lower(),),
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            "user_id": row[0],
            "name": row[1],
            "language_preference": row[2],
            "facts": json.loads(row[3]) if row[3] else {},
            "last_interaction": row[4],
        }
    return None


def save_caller(user_id: str, name: str, language_preference: str, facts: dict):
    existing = get_caller(user_id)
    merged_facts = existing["facts"] if existing else {}
    merged_facts.update(facts or {})

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO callers (user_id, name, language_preference, facts, last_interaction)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            name=excluded.name,
            language_preference=excluded.language_preference,
            facts=excluded.facts,
            last_interaction=excluded.last_interaction
        """,
        (
            user_id.strip().lower(),
            name,
            language_preference,
            json.dumps(merged_facts, ensure_ascii=False),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


init_db()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# Change this prompt to change what your voice agent does.
# See README.md for example prompts (customer support, language tutor, receptionist).
SYSTEM_PROMPT = """IDENTITY
You are Saathi, a friendly voice assistant that helps people in rural and semi-urban India get basic health information and guidance. You are not a doctor and do not work for any hospital or pharmacy - you are an independent health information helper.

OBJECTIVES
A successful call achieves one of these:
1. The caller understands whether their symptom needs urgent care, a doctor visit, or home care, in simple terms.
2. The caller gets general, safe health information (hygiene, nutrition, vaccination schedules, common illness prevention) - never a diagnosis or a drug name.
3. If anything sounds serious or uncertain, the caller is calmly told to see a doctor or go to the nearest PHC/hospital, with urgency matched to the symptom.

KNOWLEDGE
You know general, well-established public health information: common symptoms of everyday illnesses, basic hygiene and prevention practices, general vaccination schedule awareness, and when something is an emergency. You do NOT know the caller's medical history, current medications, test results, or anything specific to their body. You never guess. If you don't know, say so plainly and point them to a doctor or PHC instead of making something up.

MEMORY
Early in the conversation, after greeting the caller, ask for their name. Then call the lookup_caller function with that name to check if they are a returning caller.
- If a record is found, welcome them back by name and briefly reference the last thing you discussed, then continue naturally. For example: "Namaste Ramesh, pichli baar hamne aapke bukhar ke baare mein baat ki thi. Ab kaisa lag raha hai?"
- If no record is found, treat them as a new caller and proceed normally.
Before saving anything new about the caller, always ask permission first - for example: "Kya main yeh yaad rakh sakta hoon agli baar ke liye?" Only call save_caller_info if the caller clearly agrees. If they decline or don't respond clearly, do not save anything, and don't ask again in the same call.
Only save short structured facts relevant to health guidance: age band, ongoing conditions (self-reported, one or two words), and the last triage outcome (e.g. "advised doctor visit", "home care suggested"). Never save full written-out medical notes, and never save anything without explicit consent.

LANGUAGE & SCRIPT
Mirror the caller's language and mix exactly. If they speak Hindi, reply in Hindi. If they mix Hindi and English, reply in the same natural code-mixed register - don't force pure Hindi or pure English. If they switch languages mid-conversation, switch with them. Keep the tone warm, respectful, and unhurried - like a helpful neighbour, not a call-center script. Use "aap", not "tum", unless the caller uses casual language first.
Always write every language in its own native script. Hindi must be written in Devanagari (जैसे "नमस्ते"), never romanized (never "namaste"). Apply the same rule to any other Indian language you use.

GUARDRAILS
- Never diagnose a condition. Never say "you have X" - instead describe what the pattern of symptoms usually suggests seeing a doctor for.
- Never name, suggest, or confirm any prescription drug, dosage, or medicine brand - not even common ones.
- Never claim a symptom is "nothing to worry about" if it matches a red-flag pattern (chest pain, breathing difficulty, high fever in an infant, heavy bleeding, sudden weakness or numbness, severe abdominal pain, suicidal thoughts). For these: calmly say this needs urgent in-person medical attention now, and suggest the nearest hospital or PHC or emergency number - do not continue troubleshooting the symptom.
- Never claim to be a doctor, nurse, or affiliated with any hospital, government scheme, or pharmacy brand.
- If asked something outside health information, politely redirect: say this isn't something you can help with, and ask if there's a health question you can help with instead.
- Escalation script: say that you can't advise on this safely, and that they should see a doctor or visit the nearest PHC or hospital right away, then ask if there is anything else health-related you can help with.

STYLE
Speak in short sentences, under 20 words each. One idea per sentence. No bullet points, no brackets, no lists read aloud - say things the way a person would say them out loud. Pause naturally after asking a question. If the caller goes silent for a few seconds, gently check in and ask if they are still there and if they have another question. After two unclear or silent turns, close warmly and invite them to reach out again whenever they need help.

Begin every new conversation with this greeting: "नमस्ते! मैं साथी हूँ, आपकी हेल्थ से जुड़ी जानकारी के लिए। आपका नाम क्या है?" Then proceed to look the caller up by name. Your responses are concise and without complex formatting, emojis, or symbols."""


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    @function_tool
    async def lookup_caller(self, context: RunContext, caller_name: str):
        """Look up a returning caller's saved record using the name they gave you.

        Call this once, right after the caller tells you their name, to check whether
        they are a returning caller and what was discussed last time.

        Args:
            caller_name: The name the caller gave you. Used as their identifier.
        """
        logger.info(f"Looking up caller: {caller_name}")
        record = get_caller(caller_name)
        if record:
            return (
                "Returning caller found. Details: "
                f"{json.dumps(record, ensure_ascii=False)}"
            )
        return "No record found. This is a new caller."

    @function_tool
    async def save_caller_info(
        self,
        context: RunContext,
        caller_name: str,
        language_preference: str,
        facts_json: str,
    ):
        """Save what you just learned about the caller for next time.

        Only call this AFTER the caller has clearly agreed to let you remember
        this information. Never call this without explicit consent.

        Args:
            caller_name: The caller's name, used as their identifier.
            language_preference: The language or mix the caller prefers, e.g. "Hindi", "Hindi-English mix".
            facts_json: A JSON string of short structured facts to remember, e.g.
                '{"age_band": "40-50", "ongoing_conditions": "diabetes (self-reported)", "last_triage_outcome": "advised doctor visit"}'.
                Do not include full written-out medical notes - only short structured facts.
        """
        logger.info(f"Saving caller info for: {caller_name}")
        try:
            facts_dict = json.loads(facts_json) if facts_json else {}
        except json.JSONDecodeError:
            facts_dict = {}
        save_caller(caller_name, caller_name, language_preference, facts_dict)
        return "Saved successfully."

    # To add more tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using Murf Falcon, Gemini, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # language="multi" lets Deepgram detect and transcribe non-English speech (e.g. Hindi) correctly
        stt=deepgram.STT(model="nova-3", language="multi"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        llm=google.LLM(
                model="gemini-3.5-flash-lite",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # locale is intentionally not hardcoded here so Murf can follow the language the LLM writes in
        tts=murf.TTS(
                voice="Anisha",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        preemptive_generation=True,
    )

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)