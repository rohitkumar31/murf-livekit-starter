import json
import logging
import sqlite3
from datetime import datetime, timezone

from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    get_job_context,
    inference,
    tokenize,
    room_io,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# ---------------------------------------------------------------------------
# Day 6: outbound trunk config
# ---------------------------------------------------------------------------
OUTBOUND_TRUNK_ID = "ST_xxxxxxxx"  # get this via: lk sip outbound list

# ---------------------------------------------------------------------------
# Simple SQLite memory store for returning callers (Day 4)
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
# Day 5: domain tools
# ---------------------------------------------------------------------------
RED_FLAG_KEYWORDS = [
    "chest pain", "seene me dard", "chest me dard",
    "breathing difficulty", "saans lene me dikkat", "saans phool",
    "heavy bleeding", "bahut khoon", "zyada khoon",
    "unconscious", "behosh",
    "weakness on one side", "ek taraf kamzori", "ek taraf sunn",
    "severe stomach pain", "pet me bahut tez dard",
    "suicidal", "khud ko nuksaan",
    "infant high fever", "bache ko tez bukhar",
    "seizure", "daura",
]


def classify_triage(symptoms: str, duration_days: float) -> dict:
    text = symptoms.lower()
    for flag in RED_FLAG_KEYWORDS:
        if flag in text:
            return {
                "triage_level": "emergency",
                "reasoning": f"Description matches a known red-flag pattern ('{flag}').",
                "data_source": "Local rule-based triage guide (not a live medical data feed).",
            }
    if duration_days >= 5:
        return {
            "triage_level": "doctor_visit",
            "reasoning": "Symptom has lasted 5 or more days, which generally warrants an in-person doctor check.",
            "data_source": "Local rule-based triage guide (not a live medical data feed).",
        }
    return {
        "triage_level": "home_care",
        "reasoning": "No red-flag pattern found and duration is short. Home care and monitoring is generally reasonable, but worsening symptoms should prompt a doctor visit.",
        "data_source": "Local rule-based triage guide (not a live medical data feed).",
    }


PHC_DATASET = {
    "muzaffarpur": {"name": "Sadar Hospital, Muzaffarpur", "phone": "0621-2222083"},
    "patna": {"name": "Patna Medical College Hospital (PMCH)", "phone": "0612-2300023"},
    "vaishali": {"name": "Sadar Hospital, Hajipur, Vaishali", "phone": "06224-260204"},
    "darbhanga": {"name": "Darbhanga Medical College Hospital (DMCH)", "phone": "06272-253335"},
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """IDENTITY
You are Saathi, a friendly voice assistant that helps people in rural and semi-urban India get basic health information and guidance. You are not a doctor and do not work for any hospital or pharmacy - you are an independent health information helper.

OBJECTIVES
A successful call achieves one of these:
1. The caller understands whether their symptom needs urgent care, a doctor visit, or home care, in simple terms.
2. The caller gets general, safe health information (hygiene, nutrition, vaccination schedules, common illness prevention) - never a diagnosis or a drug name.
3. If anything sounds serious or uncertain, the caller is calmly told to see a doctor or go to the nearest PHC/hospital, with urgency matched to the symptom.

KNOWLEDGE
You know general, well-established public health information. You do NOT know the caller's medical history, current medications, test results, or anything specific to their body. You never guess. If you don't know, say so plainly.

TOOLS
When a caller describes a specific symptom and roughly how long they've had it, call check_triage_level to classify urgency instead of guessing yourself - this keeps your advice consistent. Always mention, in your own natural words, that this is based on general triage guidance, not a live medical data source.
When a caller needs to know where to physically go for care and has told you their district, call find_nearest_facility. If the tool has no facility for their district, do not invent one - tell them you don't have a specific listing for their area and advise the nearest government hospital or calling 108 for emergencies. Always be clear that this is from a small local list, not a live directory, if the caller asks or if it seems relevant.
If either tool fails or is unavailable, say so out loud in a calm, natural way and fall back to general safe advice (see a doctor or go to the nearest hospital/PHC) - never stay silent and never make up data.
If the caller wants to end the call, or opts out of future calls, call end_call.

MEMORY
Early in the conversation, after greeting the caller, ask for their name. Then call lookup_caller with that name to check if they are a returning caller.
- If found, welcome them back by name and briefly reference the last thing discussed.
- If not found, treat them as new.
Before saving anything new, always ask permission first. Only call save_caller_info if the caller clearly agrees. If they decline, do not save anything and don't ask again in the same call.
Only save short structured facts: age band, ongoing conditions (self-reported, one or two words), and last triage outcome. Never save full written-out medical notes.

LANGUAGE & SCRIPT
Mirror the caller's language and mix exactly. If they speak Hindi, reply in Hindi. If they mix Hindi and English, reply in the same natural code-mixed register. Keep the tone warm, respectful, and unhurried. Use "aap", not "tum", unless the caller uses casual language first.
Always write every language in its own native script. Hindi must be written in Devanagari (जैसे "नमस्ते"), never romanized.

GUARDRAILS
- Never diagnose a condition. Never say "you have X".
- Never name, suggest, or confirm any prescription drug, dosage, or medicine brand.
- Never claim a symptom is "nothing to worry about" if the triage tool or the pattern suggests otherwise. For emergencies: say this needs urgent in-person medical attention now, and suggest the nearest hospital/PHC or 108.
- Never claim to be a doctor, nurse, or affiliated with any hospital, government scheme, or pharmacy brand.
- If asked something outside health information, politely redirect.
- Escalation script: say you can't advise on this safely, and that they should see a doctor or visit the nearest PHC or hospital right away.

OUTBOUND CALL OPENING
If this call was placed BY YOU (an outbound follow-up call, not one the caller initiated), your very first turn must, within the first two sentences, state:
1. Who is calling and why - e.g. "नमस्ते, मैं साथी बोल रही हूँ। पिछली बार आपने जो लक्षण बताए थे, उसका फॉलो-अप करने के लिए कॉल किया है।"
2. How to opt out - e.g. "अगर आप ये कॉल्स नहीं चाहते, तो सिर्फ बोलिए 'रोक दीजिए', और मैं आगे कॉल नहीं करूँगी।"
Only after this, continue with the actual follow-up. If the caller says stop / band karo / mat karo, acknowledge respectfully and call end_call. This rule does not apply to inbound calls - for those, use the normal greeting below.

STYLE
Speak in short sentences, under 20 words each. One idea per sentence. No bullet points, no brackets, no lists read aloud. Never read raw JSON or tool output aloud - always turn it into a natural spoken sentence. Pause naturally after asking a question. If the caller goes silent, gently check in. After two unclear or silent turns, close warmly.

For INBOUND calls, begin every new conversation with this greeting: "नमस्ते! मैं साथी हूँ, आपकी हेल्थ से जुड़ी जानकारी के लिए। आपका नाम क्या है?" Then look the caller up by name. Your responses are concise and without complex formatting, emojis, or symbols."""


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    @function_tool
    async def lookup_caller(self, context: RunContext, caller_name: str):
        """Look up a returning caller's saved record using the name they gave you.

        Call this once, right after the caller tells you their name.

        Args:
            caller_name: The name the caller gave you. Used as their identifier.
        """
        logger.info(f"Looking up caller: {caller_name}")
        record = get_caller(caller_name)
        if record:
            return f"Returning caller found. Details: {json.dumps(record, ensure_ascii=False)}"
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

        Only call this AFTER the caller has clearly agreed to let you remember it.

        Args:
            caller_name: The caller's name, used as their identifier.
            language_preference: The language or mix the caller prefers.
            facts_json: A JSON string of short structured facts, e.g.
                '{"age_band": "40-50", "ongoing_conditions": "diabetes (self-reported)", "last_triage_outcome": "advised doctor visit"}'.
        """
        logger.info(f"Saving caller info for: {caller_name}")
        try:
            facts_dict = json.loads(facts_json) if facts_json else {}
        except json.JSONDecodeError:
            facts_dict = {}
        save_caller(caller_name, caller_name, language_preference, facts_dict)
        return "Saved successfully."

    @function_tool
    async def check_triage_level(
        self, context: RunContext, symptoms: str, duration_days: float
    ):
        """Classify how urgently a caller's symptoms need medical attention.

        Call this whenever a caller describes a specific symptom and how long
        they've had it, before advising home care, a doctor visit, or urgent
        care. Do not classify urgency yourself without calling this - it keeps
        advice consistent and auditable.

        Args:
            symptoms: A short description of the symptom(s) in the caller's own words.
            duration_days: Roughly how many days the caller has had the symptom.
        """
        logger.info(f"Classifying triage: symptoms={symptoms}, duration={duration_days}")
        try:
            result = classify_triage(symptoms, duration_days)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Triage classification failed: {e}")
            return (
                "TOOL_FAILED: The triage tool is unavailable right now. Tell the "
                "caller you can't assess urgency automatically at this moment, "
                "and that if the symptom feels serious or is getting worse, they "
                "should see a doctor or visit the nearest hospital or PHC to be safe."
            )

    @function_tool
    async def find_nearest_facility(self, context: RunContext, district: str):
        """Look up the nearest known government health facility for a district.

        Call this when the caller needs to know where to go for in-person care
        and has told you their district.

        Args:
            district: The caller's district name, e.g. "Muzaffarpur".
        """
        logger.info(f"Looking up facility for district={district}")
        try:
            key = district.strip().lower()
            facility = PHC_DATASET.get(key)
            if facility:
                return (
                    f"Facility found: {facility['name']}, phone {facility['phone']}. "
                    "This is from a small local sample list, not a live directory."
                )
            return (
                "NOT_FOUND: No facility listed for this district in the local "
                "sample data. Tell the caller you don't have a specific listing "
                "for their area right now, and advise the nearest government "
                "hospital or calling 108 in an emergency."
            )
        except Exception as e:
            logger.error(f"Facility lookup failed: {e}")
            return (
                "TOOL_FAILED: The facility lookup is unavailable right now. "
                "Advise the caller to go to the nearest government hospital or "
                "call 108 in an emergency."
            )

    @function_tool
    async def end_call(self, context: RunContext):
        """Call this when the caller wants to end the call, or opts out of future calls."""
        logger.info("Ending call at caller's request")
        job_ctx = get_job_context()
        if job_ctx is None:
            return
        await context.session.generate_reply(instructions="Say a brief, warm goodbye in the caller's language.")
        await job_ctx.delete_room()


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # -----------------------------------------------------------------
    # Day 6: outbound call handling
    # If job metadata has a phone_number, this is an outbound follow-up
    # call - dial out via SIP before starting the session.
    # -----------------------------------------------------------------
    dial_info = json.loads(ctx.job.metadata) if ctx.job.metadata else {}
    phone_number = dial_info.get("phone_number")
    followup_context = dial_info.get("followup_context", "")
    sip_participant_identity = phone_number

    if phone_number is not None:
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=OUTBOUND_TRUNK_ID,
                    sip_call_to=phone_number,
                    participant_identity=sip_participant_identity,
                    wait_until_answered=True,
                )
            )
            logger.info("outbound call answered")
        except api.SipCallError as e:
            logger.error(f"outbound call failed: {e.sip_status_code} {e.sip_status}")
            ctx.shutdown()
            return

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=google.LLM(
                model="gemini-3.5-flash-lite",
            ),
        tts=murf.TTS(
                voice="Anisha",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    room_options = room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=lambda params: (
                noise_cancellation.BVCTelephony()
                if params.participant.kind
                == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC()
            ),
        ),
    )

    if phone_number is not None:
        # Outbound: wait for the callee to actually join before starting the
        # session, so the greeting doesn't play into dead air.
        participant = await ctx.wait_for_participant(identity=sip_participant_identity)
        await session.start(
            agent=Assistant(),
            room=ctx.room,
            room_options=room_options,
        )
        await session.generate_reply(
            instructions=(
                "This is an outbound follow-up call you placed. "
                f"Follow-up context: {followup_context}. "
                "Deliver the compulsory opening (who's calling, why, and how to opt out) "
                "in your first two sentences, then continue with the follow-up."
            )
        )
    else:
        # Inbound: existing behaviour unchanged - caller speaks first via greeting.
        await session.start(
            agent=Assistant(),
            room=ctx.room,
            room_options=room_options,
        )
        await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)