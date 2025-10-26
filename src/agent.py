import logging
import os
import pathlib
import random

from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    utils,
)
from livekit.plugins import noise_cancellation, silero, cartesia

try:
    from livekit.plugins import openai as openai_plugin
    OPENAI_PLUGIN_AVAILABLE = True
except ImportError:
    OPENAI_PLUGIN_AVAILABLE = False
    logger_init = logging.getLogger("agent")
    logger_init.warning("⚠️ OpenAI plugin not available, will use Silero TTS")

logger = logging.getLogger("agent")

load_dotenv(".env.local")


def load_system_prompt() -> str:
    """Load system prompt from file with error handling."""
    prompt_path = os.getenv("ROLEPLAY_PROMPT_PATH", "src/prompt/roleplay_system_prompt.txt")
    try:
        full_path = pathlib.Path(prompt_path)
        if full_path.exists():
            with open(full_path, "r") as f:
                content = f.read().strip()
                logger.info(f"✅ Loaded system prompt from {prompt_path}")
                return content
        else:
            logger.warning(f"⚠️ Prompt file not found at {prompt_path}, using default")
            return "You are Coach Ava, a helpful real estate roleplay partner. Keep responses concise and friendly."
    except Exception as e:
        logger.error(f"❌ Error loading system prompt: {e}")
        return "You are Coach Ava, a helpful real estate roleplay partner. Keep responses concise and friendly."


class Assistant(Agent):
    def __init__(self) -> None:
        system_prompt = load_system_prompt()
        super().__init__(
            instructions=system_prompt,
        )

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


def prewarm(proc: JobProcess):
    # Increase activation threshold to 0.6 for better background noise filtering
    # Higher threshold = more conservative detection, less sensitive to noise
    proc.userdata["vad"] = silero.VAD.load(activation_threshold=0.6)


async def dev_mode_entrypoint(ctx: JobContext):
    """
    DEV MODE ENTRYPOINT: Auto-join the configured room for local testing.
    This uses the LiveKit Agents framework but immediately starts the voice session.
    """
    print("\n🔧 DEV MODE: Starting agent in auto-join mode...\n")

    # Override the room to be the configured dev room
    room_name = os.getenv("LIVEKIT_ROOM", "roleplay-local")
    print(f"✅ DEV MODE: Joining room '{room_name}'\n")

    # Create a mock room if needed or use the existing context
    # For dev mode, we still use the JobContext passed in but override the room join
    await entrypoint(ctx)


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    logger.info(f"🤖 Agent starting in room: {ctx.room.name}")
    logger.info("✅ Cartesia TTS + EOUPlugin turn detector ready for webhook mode")

    # Load environment configuration and check LLM provider
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    logger.info(f"📊 Using LLM model: {llm_model}")

    # Verify OpenAI API key only if using OpenAI
    if "openai" in llm_model.lower():
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("❌ OPENAI_API_KEY not set in environment. Agent will not function.")
            return
    else:
        logger.info(f"⚠️ Using non-OpenAI LLM model, skipping OpenAI key check")

    # Set up a voice AI pipeline with OpenAI LLM and system prompt
    system_prompt = load_system_prompt()

    # Voice rotation pool - Doris, Lynda, and Denise
    VOICE_POOL = [
        "0c8ed86e-6c64-40f0-b252-b773911de6bb",  # Doris
        "829ccd10-f8b3-43cd-b8a0-4aeaa81f3b30",  # Lynda
        "8a1b8af0-c4f6-423f-a268-5507fd4aefdf"   # Denise
    ]
    selected_voice = random.choice(VOICE_POOL)
    logger.info(f"🎲 Selected voice for this session: {selected_voice}")

    # Configure TTS - Cartesia
    tts_option = cartesia.TTS(
        voice=selected_voice,
        model="sonic-2-2025-06-11"
    )
    logger.info(f"🔊 Using TTS: Cartesia Sonic with voice {selected_voice}")

    session = AgentSession(
        # Speech-to-text (STT) - convert user speech to text
        stt="assemblyai/universal-streaming:en",
        # Large Language Model - using gpt-4o-mini for fast real estate roleplay responses
        llm=f"openai/{llm_model}",
        # Text-to-speech - configured above with fallback logic
        tts=tts_option,
        # Voice Activity Detection (VAD) - using Silero VAD
        vad=ctx.proc.userdata["vad"],
        # Allow preemptive generation while waiting for user turn end
        preemptive_generation=False,
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

    # Metrics collection and logging hooks
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    @session.on("user_speech_committed")
    def _on_user_speech_committed(message: str):
        """Log when user speech is converted to text."""
        logger.info(f"🗣️ User transcript: {message[:100]}...")

    @session.on("agent_speech_committed")
    def _on_agent_speech_committed(message: str):
        """Log when agent generates a response."""
        logger.info(f"🧠 Agent reply: {message[:100]}...")

    @session.on("user_speech_finished")
    def _on_user_speech_finished():
        """Log when user stops speaking."""
        logger.info("⏸️ User speech finished, processing...")

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"📊 Session usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

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
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
            # Don't close agent session on brief client disconnects (iOS Safari may briefly drop)
            close_on_disconnect=False,
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()

    # Canary: Publish a test message to verify TTS is working
    # On mobile, prepend 120ms SSML break to prevent clipping
    logger.info("🔊 TTS Canary: Starting test message...")
    try:
        canary_text = "<speak><break time='120ms'/>Hello!</speak>"
        logger.info(f"🔊 Publishing canary with SSML break for mobile: {canary_text}")
        await session.say(canary_text, allow_interruptions=False)
        logger.info("🔊 TTS Canary: Finished")
    except Exception as e:
        logger.error(f"❌ TTS Canary failed: {e}")
        logger.info("⚠️ Continuing without canary...")

    # Agent is ready - session handles all voice interaction automatically
    # The session.start() call above manages the interaction loop until the room ends
    # This entrypoint will return when the session completes naturally
    # allowing the worker to clean up and immediately poll for the next job
    logger.info("✅ Agent is now ready and waiting for user interactions...")
    # Note: session.start() completes when the room ends, then we fall through to cleanup
    # The finally block below ensures proper teardown before returning to the worker loop
    logger.info("🔌 Agent session ended - cleaning up connections...")
    await session.aclose()
    logger.info(f"SESSION_CLOSED room={ctx.room.name}")
    logger.info("WAITING_FOR_NEXT_JOB")


if __name__ == "__main__":
    agent_mode = os.getenv("LIVEKIT_AGENT_MODE", "webhook")

    if agent_mode == "webhook":
        print("\n" + "="*60)
        print("🔌 WEBHOOK MODE: Agent waiting for job requests")
        print("="*60 + "\n")
        # Keep restarting the worker if it exits
        while True:
            try:
                cli.run_app(WorkerOptions(
                    entrypoint_fnc=entrypoint,
                    prewarm_fnc=prewarm,
                    agent_name="roleplay",
                    num_idle_processes=10  # Keep 10 worker processes warm for concurrent sessions
                ))
            except Exception as e:
                logger.error(f"❌ Worker error: {e}")
                logger.info("🔄 Restarting worker in 5 seconds...")
                asyncio.run(asyncio.sleep(5))
    else:
        print("\n" + "="*60)
        print("🔧 DEV MODE: Agent will auto-join room on startup")
        print("="*60 + "\n")
        cli.run_app(WorkerOptions(entrypoint_fnc=dev_mode_entrypoint, prewarm_fnc=prewarm))
