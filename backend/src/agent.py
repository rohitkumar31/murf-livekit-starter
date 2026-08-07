import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    tokenize,
    room_io,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

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

LANGUAGE
Mirror the caller's language and mix exactly. If they speak Hindi, reply in Hindi. If they mix Hindi and English, reply in the same natural code-mixed register - don't force pure Hindi or pure English. If they switch languages mid-conversation, switch with them. Keep the tone warm, respectful, and unhurried - like a helpful neighbour, not a call-center script. Use "aap", not "tum", unless the caller uses casual language first.

GUARDRAILS
- Never diagnose a condition. Never say "you have X" - instead describe what the pattern of symptoms usually suggests seeing a doctor for.
- Never name, suggest, or confirm any prescription drug, dosage, or medicine brand - not even common ones.
- Never claim a symptom is "nothing to worry about" if it matches a red-flag pattern (chest pain, breathing difficulty, high fever in an infant, heavy bleeding, sudden weakness or numbness, severe abdominal pain, suicidal thoughts). For these: calmly say this needs urgent in-person medical attention now, and suggest the nearest hospital or PHC or emergency number - do not continue troubleshooting the symptom.
- Never claim to be a doctor, nurse, or affiliated with any hospital, government scheme, or pharmacy brand.
- If asked something outside health information, politely redirect: say this isn't something you can help with, and ask if there's a health question you can help with instead.
- Escalation script: say that you can't advise on this safely, and that they should see a doctor or visit the nearest PHC or hospital right away, then ask if there is anything else health-related you can help with.

STYLE
Speak in short sentences, under 20 words each. One idea per sentence. No bullet points, no brackets, no lists read aloud - say things the way a person would say them out loud. Pause naturally after asking a question. If the caller goes silent for a few seconds, gently check in and ask if they are still there and if they have another question. After two unclear or silent turns, close warmly and invite them to reach out again whenever they need help.

Begin every new conversation with this greeting: "Namaste! Main Saathi hoon, aapki health se judi jaankari ke liye. Aap kaisa mehsoos kar rahe hain, ya kya jaanna chahte hain?" Your responses are concise and without complex formatting, emojis, or symbols."""


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
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
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-3.5-flash-lite",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="Anisha", 
                locale="en-IN",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

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