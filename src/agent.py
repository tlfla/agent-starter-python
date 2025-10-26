import logging
import os
import pathlib
import random
import json
from datetime import datetime, timezone

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

# Import OpenAI for evaluation API calls
try:
    import openai
    OPENAI_API_AVAILABLE = True
except ImportError:
    OPENAI_API_AVAILABLE = False
    openai = None

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class TranscriptCollector:
    """Collects conversation transcript from multiple sources."""
    def __init__(self):
        self.items = []

    def note_user(self, text: str):
        if text.strip():
            self.items.append({
                "t": datetime.now(timezone.utc).isoformat(),
                "speaker": "user",
                "text": text
            })

    def note_agent(self, text: str):
        if text.strip():
            self.items.append({
                "t": datetime.now(timezone.utc).isoformat(),
                "speaker": "agent",
                "text": text
            })

    def to_json(self) -> list:
        return self.items

    def __len__(self):
        return len(self.items)


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
    def __init__(self, transcript_collector: TranscriptCollector) -> None:
        system_prompt = load_system_prompt()
        self.transcript_collector = transcript_collector
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


async def evaluate_call(transcript_buffer: list, openai_api_key: str) -> dict:
    """
    Send transcript to OpenAI for comprehensive evaluation using full coaching prompt.
    Returns detailed evaluation with scores, wins, improvements, and training links.
    """
    logger.info(f"📊 evaluate_call started with {len(transcript_buffer)} transcript items")

    if not OPENAI_API_AVAILABLE or not openai:
        logger.error("❌ OpenAI not available")
        return {"error": "OpenAI not available"}

    if not transcript_buffer:
        logger.error("❌ No transcript data")
        return {"error": "No transcript data"}

    try:
        # Training mapping for lesson recommendations
        training_mapping = {
            "rapport_building": "Elicitation Module 1: Intent, Rapport, Psychological Safety",
            "objection_handling": "Elicitation Module 7: Resistance & De-escalation",
            "tone_confidence": "Negotiation Module 4: Late-Night FM DJ Voice",
            "mirror_matching": "Elicitation Module 3: Pace Mirror",
            "vak_alignment": "DISC/VAK: VAK Learning Styles"
        }

        # Simplified evaluation prompt focused on core metrics
        evaluation_prompt = """You are an expert real estate coach evaluating a completed roleplay call.

SCORING (0-10 scale):
- 9-10: Exceptional mastery
- 7-8: Strong execution
- 5-6: Adequate but inconsistent
- 3-4: Needs work
- 0-2: Poor or harmful

REQUIRED OUTPUT (valid JSON only):
{
  "overall_score": 7.8,
  "scores": {
    "rapport_building": 8,
    "objection_handling": 7,
    "tone_confidence": 8
  },
  "top_wins": [
    "Strong warm introduction that created psychological safety",
    "Used Name-Pace-Bridge technique on commission objection",
    "Maintained confident tone with downward inflection"
  ],
  "top_improvements": [
    "Label emotions before proceeding ('It sounds like you're feeling uncertain')",
    "Match client's slower pace during objections instead of speeding up",
    "Use embedded commands ('Tuesday or Thursday?' vs 'Can we meet?')"
  ],
  "training_links": [
    "Elicitation Module 4: Acknowledge and Validate",
    "Elicitation Module 3: Pace Mirror"
  ],
  "summary": "Strong rapport and tone. Focus on labeling emotions explicitly and matching client pace during resistance."
}

Return ONLY valid JSON. Be specific and encouraging."""

        # Format transcript for evaluation
        formatted_transcript = []
        for entry in transcript_buffer:
            speaker = "Realtor" if entry["speaker"] == "agent" else "Client"
            formatted_transcript.append(f"{speaker}: {entry['text']}")

        transcript_text = "\n".join(formatted_transcript)
        logger.info(f"📝 Formatted transcript: {len(transcript_text)} chars")

        # Call OpenAI with full evaluation
        logger.info("🔄 Creating OpenAI client and calling API...")
        client = openai.OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": evaluation_prompt},
                {"role": "user", "content": f"Evaluate this roleplay call:\n\n{transcript_text}"}
            ],
            temperature=0.3,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        logger.info("✅ OpenAI API call completed")

        # Parse response
        result_text = response.choices[0].message.content.strip()
        logger.info(f"📄 Response text: {len(result_text)} chars")
        result = json.loads(result_text)
        logger.info(f"✅ Parsed JSON successfully: {list(result.keys())}")

        return result

    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parse error: {e}")
        logger.error(f"Raw response: {result_text if 'result_text' in locals() else 'N/A'}")
        return {"error": "Invalid JSON from evaluator"}
    except Exception as e:
        logger.error(f"❌ Evaluation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


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

    # Transcript collector for evaluation
    transcript_collector = TranscriptCollector()
    evaluate_enabled = False

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

    # Voice rotation pool - Doris and 2 new voices
    VOICE_POOL = [
        "0c8ed86e-6c64-40f0-b252-b773911de6bb",  # Doris
        "78ab82d5-25be-4f7d-82b3-7ad64e5b85b2",  # New voice 1
        "66c6b81c-ddb7-4892-bdd5-19b5a7be38e7"   # New voice 2
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

    # Primary user transcript capture - fires when STT finalizes transcription
    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev):
        """Capture user speech when transcription is finalized."""
        if hasattr(ev, 'is_final') and ev.is_final:
            transcript_text = ev.transcript if hasattr(ev, 'transcript') else str(ev)
            logger.info(f"🗣️ User transcript (final): {transcript_text[:100]}...")
            transcript_collector.note_user(transcript_text)
            logger.info(f"📝 Transcript has {len(transcript_collector)} items")

    # Legacy event handlers (may not fire in SDK 1.2.15, but kept for compatibility)
    @session.on("user_speech_committed")
    def _on_user_speech_committed(message: str):
        """Fallback for user speech (legacy event)."""
        logger.info(f"🗣️ User transcript (legacy): {message[:100]}...")
        transcript_collector.note_user(message)
        logger.info(f"📝 Transcript has {len(transcript_collector)} items")

    @session.on("agent_speech_committed")
    def _on_agent_speech_committed(message: str):
        """Capture agent final reply text."""
        logger.info(f"🧠 Agent reply: {message[:100]}...")
        transcript_collector.note_agent(message)
        logger.info(f"📝 Transcript has {len(transcript_collector)} items")

    @session.on("user_speech_finished")
    def _on_user_speech_finished():
        """Log when user stops speaking."""
        logger.info("⏸️ User speech finished, processing...")

    # Data channel handler to receive evaluate flag from frontend
    @ctx.room.on("data_received")
    def _on_data_received(data_packet: rtc.DataPacket):
        """Handle data messages from frontend (e.g., evaluate flag)."""
        nonlocal evaluate_enabled
        try:
            payload = json.loads(data_packet.data.decode("utf-8"))
            if payload.get("type") == "evaluate":
                evaluate_enabled = payload.get("value", False)
                logger.info(f"📊 Evaluation {'enabled' if evaluate_enabled else 'disabled'} by client")
            elif payload.get("type") == "request_evaluation":
                # Frontend is about to disconnect, run evaluation now
                logger.info("📊 Received evaluation request from client")
                # Run evaluation asynchronously without blocking
                import asyncio
                asyncio.create_task(run_evaluation())
        except Exception as e:
            logger.error(f"Error parsing data message: {e}")

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"📊 Session usage: {summary}")

    async def run_evaluation():
        """Run evaluation if enabled and send result via data channel."""
        try:
            logger.info(f"🔍 run_evaluation called, evaluate_enabled={evaluate_enabled}")

            if not evaluate_enabled:
                logger.info("Evaluation not enabled, skipping")
                return

            # Check if we have transcript data
            transcript_items = transcript_collector.to_json()
            user_turns = [t for t in transcript_items if t["speaker"] == "user"]

            if not user_turns:
                logger.warning(f"No user speech to evaluate (collector has {len(transcript_collector)} items)")
                # Send error response
                try:
                    error_result = {"type": "evaluation_ready", "data": {"error": "no_transcript", "summary": ""}}
                    error_json = json.dumps(error_result)
                    await ctx.room.local_participant.publish_data(
                        error_json.encode("utf-8"),
                        reliable=True
                    )
                    logger.info(f"evaluation_ready: {len(error_json)} chars")
                except Exception as e:
                    logger.error(f"Failed to send error result: {e}")
                return

            logger.info(f"✅ Starting evaluation on {len(transcript_collector)} transcript items...")

            # Get OpenAI API key
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error("❌ OPENAI_API_KEY not set, cannot evaluate")
                return

            # Run evaluation
            logger.info("🤖 Calling OpenAI for evaluation...")
            result = await evaluate_call(transcript_items, openai_api_key)
            logger.info(f"✅ OpenAI returned result: {list(result.keys())}")

            # Send result via data channel
            try:
                result_json = json.dumps({"type": "evaluation_ready", "data": result})
                await ctx.room.local_participant.publish_data(
                    result_json.encode("utf-8"),
                    reliable=True
                )
                logger.info(f"evaluation_ready: {len(result_json)} chars")
            except Exception as e:
                logger.error(f"❌ Failed to send evaluation result: {e}")
                import traceback
                logger.error(traceback.format_exc())
        except Exception as e:
            logger.error(f"❌ Error in run_evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())

    ctx.add_shutdown_callback(log_usage)
    # Note: run_evaluation is now called via data channel message, not shutdown callback

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(transcript_collector),
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

    # Simple greeting and wait for session to complete
    # This blocks until the room disconnects or session ends
    logger.info("✅ Agent ready - saying hello and waiting for user...")
    await session.generate_reply(
        instructions="Say only the word 'Hello' in a friendly tone. Do not say anything else."
    )


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
