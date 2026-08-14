import json
import logging
import os
import sqlite3
import urllib.request
import uuid
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

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS escalations (
            reference_id TEXT PRIMARY KEY,
            caller_name TEXT,
            reason TEXT,
            summary TEXT,
            what_was_checked TEXT,
            urgency TEXT,
            language TEXT,
            follow_up_method TEXT,
            status TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS calls (
            call_id TEXT PRIMARY KEY,
            outcome TEXT,
            channel TEXT,
            created_at TEXT
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


def create_escalation_record(
    caller_name, reason, summary, what_was_checked, urgency, language, follow_up_method
):
    reference_id = "ESC-" + uuid.uuid4().hex[:6].upper()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO escalations
        (reference_id, caller_name, reason, summary, what_was_checked, urgency, language, follow_up_method, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            reference_id,
            caller_name,
            reason,
            summary,
            what_was_checked,
            urgency,
            language,
            follow_up_method,
            "open",
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return reference_id


def send_to_discord(reference_id, caller_name, reason, summary, what_was_checked, urgency, language, follow_up_method):
    if not DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL not set, skipping webhook send")
        return False
    content = (
        f"**New Escalation - {reference_id}**\n"
        f"**Urgency:** {urgency}\n"
        f"**Caller:** {caller_name}\n"
        f"**Reason:** {reason}\n"
        f"**Summary:** {summary}\n"
        f"**Already checked:** {what_was_checked}\n"
        f"**Language:** {language}\n"
        f"**Follow-up method:** {follow_up_method}\n"
    )
    payload = json.dumps({"content": content}).encode("utf-8")
    req = urllib.request.Request(
        DISCORD_WEBHOOK_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        logger.error(f"Discord webhook send failed: {e}")
        return False


def record_call(call_id: str, outcome: str, channel: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO calls (call_id, outcome, channel, created_at) VALUES (?, ?, ?, ?)",
        (call_id, outcome, channel, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


init_db()

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
        "reasoning": "No red-flag pattern found and duration is short.",
        "data_source": "Local rule-based triage guide (not a live medical data feed).",
    }


PHC_DATASET = {
    "muzaffarpur": {"name": "Sadar Hospital, Muzaffarpur", "phone": "0621-2222083", "opd_hours": "9 AM - 2 PM, Mon-Sat", "walk_in": True},
    "patna": {"name": "Patna Medical College Hospital (PMCH)", "phone": "0612-2300023", "opd_hours": "8 AM - 1 PM, Mon-Sat", "walk_in": True},
    "vaishali": {"name": "Sadar Hospital, Hajipur, Vaishali", "phone": "06224-260204", "opd_hours": "9 AM - 2 PM, Mon-Sat", "walk_in": True},
    "darbhanga": {"name": "Darbhanga Medical College Hospital (DMCH)", "phone": "06272-253335", "opd_hours": "8 AM - 1 PM, Mon-Sat", "walk_in": True},
}

MAIN_SYSTEM_PROMPT = """IDENTITY
You are Saathi, a friendly voice assistant that helps people in rural and semi-urban India get basic health information and guidance. You are not a doctor and do not work for any hospital or pharmacy - you are an independent health information helper.

OBJECTIVES
A successful call achieves one of these:
1. The caller understands whether their symptom needs urgent care, a doctor visit, or home care, in simple terms.
2. The caller gets general, safe health information - never a diagnosis or a drug name.
3. If anything sounds serious or uncertain, the caller is calmly told to see a doctor or go to the nearest PHC/hospital.

KNOWLEDGE
You know general, well-established public health information. You do NOT know the caller's medical history or anything specific to their body. You never guess.

TOOLS
When a caller describes a specific symptom and roughly how long they've had it, call check_triage_level to classify urgency instead of guessing yourself. Always mention this is based on general triage guidance, not a live medical data source.
If it fails, say so out loud calmly and fall back to general safe advice - never stay silent, never make up data.

HANDOFF TO CLINIC & APPOINTMENT SPECIALIST
You are a general health guidance assistant, not a booking or logistics assistant. If the caller asks specifically about which clinic/hospital to visit, opening hours, whether they need an appointment or can walk in, or how to reach a facility, do NOT try to answer this yourself - call transfer_to_clinic_specialist instead. Before transferring, tell the caller clearly and simply, e.g. "Main aapko hamare clinic specialist se connect karta hoon." Only transfer for clinic/appointment/visiting-logistics questions - not for triage, general health info, or escalation situations.

ESCALATION TO A HUMAN
There are exactly two situations where you must offer to escalate to a human:
1. The caller describes a red-flag symptom.
2. The caller directly asks you to diagnose them or name what disease/condition they have.
Tell the caller plainly this is beyond what you can safely help with, explain what information you'd like to send, and ask permission clearly. If they say no, give standard safety advice instead.
If they agree, call create_escalation with a short, factual summary. Never include OTPs, PINs, passwords, or account numbers. After success, tell the caller their reference ID and an honest next step.

MEMORY
Early in the conversation, after greeting the caller, ask for their name. Then call lookup_caller with that name.
- If found, welcome them back by name and briefly reference the last thing discussed.
- If not found, treat them as new.
Before saving anything new, always ask permission first. Only call save_caller_info if the caller clearly agrees.
Only save short structured facts: age band, ongoing conditions (self-reported, one or two words), and last triage outcome.

LANGUAGE & SCRIPT
Mirror the caller's language and mix exactly. Keep the tone warm, respectful, and unhurried. Use "aap", not "tum", unless the caller uses casual language first.
Always write every language in its own native script. Hindi must be written in Devanagari (जैसे "नमस्ते"), never romanized.

GUARDRAILS
- Never diagnose a condition. Never name or suggest any prescription drug or dosage.
- Never claim to be a doctor, nurse, or affiliated with any hospital, government scheme, or pharmacy brand.
- If asked something outside health information, politely redirect.

STYLE
Speak in short sentences, under 20 words each. One idea per sentence. No bullet points, no brackets, no lists read aloud. Never read raw JSON or tool output aloud. If the caller goes silent, gently check in.

Begin every new conversation with this greeting: "नमस्ते! मैं साथी हूँ, आपकी हेल्थ से जुड़ी जानकारी के लिए। आपका नाम क्या है?" Then look the caller up by name. Your responses are concise and without complex formatting, emojis, or symbols."""


SPECIALIST_SYSTEM_PROMPT = """IDENTITY
You are the Clinic & Appointment Specialist for Saathi. You have just taken over the conversation from Saathi, the general health assistant. You only handle one job: helping the caller figure out which facility to visit, its hours, and whether they need to book or can walk in.

FIRST TURN
Start by briefly introducing yourself, e.g. "नमस्ते, मैं क्लिनिक विशेषज्ञ हूँ। मैं आपको सही जगह और समय बताने में मदद करूँगा।" Do not make the caller repeat their question if they already told Saathi what they need - continue naturally from the conversation so far.

KNOWLEDGE
You know a small local list of government facilities with their OPD hours and whether walk-ins are accepted. This is a hand-built local sample list, not a live directory - mention this naturally if relevant.

TOOLS
Call find_facility_details when the caller has told you (or told Saathi) their district. If no match, say so honestly and suggest the nearest government hospital or calling 108 for emergencies - never invent a facility.

LANGUAGE & SCRIPT
Mirror the caller's language and mix. Hindi must be written in Devanagari, never romanized.

LIMITS
You do not give medical guidance, triage, or diagnoses - if the caller brings up a new symptom question, gently say that's outside what you handle and that Saathi can help with that. Keep answers short, spoken, and practical - hours, walk-in vs appointment, and how to reach the facility.

STYLE
Short sentences, under 20 words each. No bullet points, no brackets, no lists read aloud."""


class ClinicSpecialist(Agent):
    def __init__(self, main_agent_ref=None, caller_name: str = "") -> None:
        super().__init__(instructions=SPECIALIST_SYSTEM_PROMPT)
        self.main_agent_ref = main_agent_ref
        self.caller_name = caller_name

    @function_tool
    async def find_facility_details(self, context: RunContext, district: str):
        """Look up a facility's name, phone, OPD hours, and walk-in policy for a district.

        Args:
            district: The caller's district name, e.g. "Muzaffarpur".
        """
        try:
            key = district.strip().lower()
            facility = PHC_DATASET.get(key)
            if facility:
                if self.main_agent_ref:
                    self.main_agent_ref.call_succeeded = True
                walkin = "walk-ins are accepted" if facility["walk_in"] else "an appointment is needed"
                return (
                    f"{facility['name']}, phone {facility['phone']}. "
                    f"OPD hours: {facility['opd_hours']}. {walkin}. "
                    "This is from a small local sample list, not a live directory."
                )
            return "NOT_FOUND: advise the nearest government hospital or calling 108."
        except Exception as e:
            logger.error(f"Facility detail lookup failed: {e}")
            return "TOOL_FAILED: advise the nearest government hospital or calling 108."


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=MAIN_SYSTEM_PROMPT)
        self.call_succeeded = False
        self.caller_name = ""

    @function_tool
    async def lookup_caller(self, context: RunContext, caller_name: str):
        """Look up a returning caller's saved record using the name they gave you.

        Args:
            caller_name: The name the caller gave you. Used as their identifier.
        """
        self.caller_name = caller_name
        record = get_caller(caller_name)
        if record:
            return f"Returning caller found. Details: {json.dumps(record, ensure_ascii=False)}"
        return "No record found. This is a new caller."

    @function_tool
    async def save_caller_info(
        self, context: RunContext, caller_name: str, language_preference: str, facts_json: str
    ):
        """Save what you just learned about the caller for next time. Only call after explicit consent.

        Args:
            caller_name: The caller's name, used as their identifier.
            language_preference: The language or mix the caller prefers.
            facts_json: A JSON string of short structured facts.
        """
        try:
            facts_dict = json.loads(facts_json) if facts_json else {}
        except json.JSONDecodeError:
            facts_dict = {}
        save_caller(caller_name, caller_name, language_preference, facts_dict)
        return "Saved successfully."

    @function_tool
    async def check_triage_level(self, context: RunContext, symptoms: str, duration_days: float):
        """Classify how urgently a caller's symptoms need medical attention.

        Args:
            symptoms: A short description of the symptom(s) in the caller's own words.
            duration_days: Roughly how many days the caller has had the symptom.
        """
        try:
            result = classify_triage(symptoms, duration_days)
            self.call_succeeded = True
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Triage classification failed: {e}")
            return "TOOL_FAILED: advise the caller to see a doctor or visit the nearest hospital/PHC to be safe."

    @function_tool
    async def create_escalation(
        self,
        context: RunContext,
        caller_name: str,
        reason: str,
        summary: str,
        what_was_checked: str,
        urgency: str,
        language: str,
        follow_up_method: str,
    ):
        """Create a request for a human health worker to follow up. Only call this
        AFTER the caller has clearly given permission to share their information.

        Args:
            caller_name: The caller's name.
            reason: Short reason for escalation.
            summary: A short, factual summary. Never include OTPs, PINs, passwords, or account numbers.
            what_was_checked: What you (the agent) already checked or told the caller.
            urgency: One of "low", "medium", "high", "emergency".
            language: The caller's language or mix.
            follow_up_method: How the caller prefers to be contacted back.
        """
        logger.info(f"Creating escalation for {caller_name}: {reason}")
        try:
            reference_id = create_escalation_record(
                caller_name, reason, summary, what_was_checked, urgency, language, follow_up_method
            )
            send_to_discord(
                reference_id, caller_name, reason, summary, what_was_checked, urgency, language, follow_up_method
            )
            self.call_succeeded = True
            return f"Escalation created successfully. Reference ID: {reference_id}"
        except Exception as e:
            logger.error(f"create_escalation failed: {e}")
            return "TOOL_FAILED: tell the caller you couldn't create the request right now, and advise them to see a doctor or go to the nearest hospital/PHC directly."

    @function_tool
    async def transfer_to_clinic_specialist(self, context: RunContext, reason: str):
        """Hand off the conversation to the Clinic & Appointment Specialist.

        Call this ONLY when the caller asks which facility to visit, its opening
        hours, whether they need an appointment or can walk in, or how to reach
        it. Do not call this for triage, general health questions, or escalation
        situations - those stay with you. Tell the caller you're connecting them
        to the specialist before calling this.

        Args:
            reason: Brief reason for the handoff, e.g. "asked about clinic hours".
        """
        logger.info(f"Handing off to clinic specialist: {reason}")
        return ClinicSpecialist(main_agent_ref=self, caller_name=self.caller_name)


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

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

    assistant = Assistant()
    call_id = f"{ctx.room.name}-{uuid.uuid4().hex[:6]}"

    async def on_shutdown():
        outcome = "success" if assistant.call_succeeded else "failed"
        try:
            record_call(call_id, outcome, "browser")
            logger.info(f"Recorded call {call_id} as {outcome}")
        except Exception as e:
            logger.error(f"Failed to record call outcome: {e}")

    ctx.add_shutdown_callback(on_shutdown)

    await session.start(
        agent=assistant,
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

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)