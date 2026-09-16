import os

os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import sys
import time
import uuid
import json
import re
import math
import numpy as np
import base64
import queue
import threading
import asyncio
import logging
from typing import Optional
import wave

import aiohttp
import torch
import sentencepiece
import scipy.signal as signal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

try:
    from moshi.models import loaders, MimiModel, LMModel
    from chatterbox.tts import ChatterboxTTS
except ImportError:
    print("Error: Moshi models not found. Please install required dependencies.")
    sys.exit(1)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('audio_server.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
SESSION_DIR = "session_storage"
os.makedirs(SESSION_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-ID"]
)


# --- Session Management ---
def load_session(session_id: str) -> dict:
    session_path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if os.path.exists(session_path):
        try:
            with open(session_path, "r") as f:
                data = json.load(f)
                logger.debug(f"📂 Loaded session {session_id} with {len(data.get('messages', []))} messages")
                return data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load session {session_id}: {e}")
            return {"messages": []}
    logger.debug(f"📂 Creating new session {session_id}")
    return {"messages": []}


def save_session(session_id: str, session_data: dict):
    session_path = os.path.join(SESSION_DIR, f"{session_id}.json")
    try:
        with open(session_path, "w") as f:
            json.dump(session_data, f, indent=4)
        logger.debug(f"💾 Saved session {session_id} with {len(session_data.get('messages', []))} messages")
    except Exception as e:
        logger.error(f"❌ Failed to save session {session_id}: {e}")


# --- STT Inference State ---
from dataclasses import dataclass


@dataclass
class InferenceState:
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm: LMModel
    lm_gen: any
    device: str | torch.device
    frame_size: int
    batch_size: int
    accumulated_text: str
    accumulated_token_ids: list
    # CRITICAL: STT config parameters from model
    audio_silence_prefix_seconds: float
    audio_delay_seconds: float
    padding_token_id: int
    # Prefix tracking
    n_prefix_chunks: int
    prefix_chunks_fed: int
    is_in_prefix_phase: bool

    def __init__(self, 
                mimi: MimiModel, 
                text_tokenizer: sentencepiece.SentencePieceProcessor,
                lm: LMModel, 
                batch_size: int, 
                device: str | torch.device,
                stt_config: dict, raw_config: dict):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.lm = lm
        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.batch_size = batch_size
        
        # Load CRITICAL STT configuration for VAD with proper defaults
        # If config is empty or None, use reference implementation defaults
        if not stt_config or not isinstance(stt_config, dict):
            logger.warning("⚠️ STT config is empty or invalid, using default values")
            stt_config = {}
        
        self.audio_silence_prefix_seconds = stt_config.get("audio_silence_prefix_seconds", 1.0)
        self.audio_delay_seconds = stt_config.get("audio_delay_seconds", 5.0)
        self.padding_token_id = raw_config.get("text_padding_token_id", 3)
        
        # FORCE minimum values if config returns 0 or invalid values
        if self.audio_silence_prefix_seconds <= 0:
            logger.warning(f"⚠️ Invalid audio_silence_prefix_seconds: {self.audio_silence_prefix_seconds}, forcing to 1.0s")
            self.audio_silence_prefix_seconds = 1.0
        
        if self.audio_delay_seconds <= 0:
            logger.warning(f"⚠️ Invalid audio_delay_seconds: {self.audio_delay_seconds}, forcing to 5.0s")
            self.audio_delay_seconds = 5.0
        
        # Calculate prefix chunks (CRITICAL for VAD calibration)
        self.n_prefix_chunks = max(1, math.ceil(self.audio_silence_prefix_seconds * self.mimi.frame_rate))
        
        self.reset()
        logger.info(f"✅ STT InferenceState created - frame_size: {self.frame_size}, frame_rate: {self.mimi.frame_rate}, "
                   f"silence_prefix: {self.audio_silence_prefix_seconds}s ({self.n_prefix_chunks} chunks), "
                   f"audio_delay: {self.audio_delay_seconds}s")

    def reset(self):
        from moshi.models import LMGen
        logger.info("🔄 Resetting STT utterance state for next turn.")
        self.lm_gen = LMGen(self.lm, temp=0, temp_text=0, use_sampling=False)
        self.accumulated_text = ""
        self.accumulated_token_ids = []
        # Reset prefix tracking
        self.prefix_chunks_fed = 0
        self.is_in_prefix_phase = True


# --- Model Manager ---
class ModelManager:
    """Singleton to manage STT and TTS models."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stt_mimi = None
        self.stt_text_tokenizer = None
        self.stt_lm = None
        self.stt_checkpoint_info = None
        self.tts_model = None
        self.tts_sample_rate = 24000
        self._load_models()
        self._initialized = True

    def _load_models(self):
        try:
            logger.info("🔧 Initializing STT model...")
            self.stt_checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr-candle")
            self.stt_mimi = self.stt_checkpoint_info.get_mimi(device=self.device)
            self.stt_text_tokenizer = self.stt_checkpoint_info.get_text_tokenizer()
            self.stt_lm = self.stt_checkpoint_info.get_moshi(device=self.device)
            
            # Log the loaded STT config
            logger.info(f"📋 STT Config: {self.stt_checkpoint_info.stt_config}")
            logger.info(f"📋 Raw Config (text_padding_token_id): {self.stt_checkpoint_info.raw_config.get('text_padding_token_id')}")
            logger.info("✅ STT model initialized successfully.")

            logger.info("🔧 Initializing Chatterbox TTS model...")
            self.tts_model = ChatterboxTTS.from_pretrained(device="cuda")
            self.tts_sample_rate = self.tts_model.sr
            logger.info(f"✅ Chatterbox TTS model initialized - sample_rate: {self.tts_sample_rate}")
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}", exc_info=True)
            sys.exit(1)


model_manager = ModelManager()


# --- TTS Streaming Handler ---
class TTSStreamingHandler:
    """Handles TTS generation with real-time streaming."""

    def __init__(self, tts_model: ChatterboxTTS):
        self.tts_model = tts_model
        self.sample_rate = tts_model.sr

    def audio_to_base64_pcm(self, audio_chunk: torch.Tensor) -> str:
        """Convert torch tensor audio chunk to base64 PCM."""
        audio_np = audio_chunk.cpu().numpy()
        audio_int16 = (np.clip(audio_np, -1, 1) * 32767).astype(np.int16)
        return base64.b64encode(audio_int16.tobytes()).decode('utf-8')

    async def generate_and_stream(self, websocket: WebSocket, text: str):
        """Generate TTS audio and stream it in real-time to the websocket."""
        
        logger.info(f"🎵 Generating Chatterbox TTS for: '{text[:50]}...'")
        chunk_count = 0

        try:
            queue_out = queue.Queue()
            finished_event = threading.Event()

            def _generate():
                try:
                    for audio_chunk, metrics in self.tts_model.generate_stream(text):
                        queue_out.put(audio_chunk)
                except Exception as e:
                     logger.error(f"❌ TTS generation error: {e}", exc_info=True)
                finally:
                    finished_event.set()

            # Start generation in a separate thread
            threading.Thread(target=_generate, daemon=True).start()

            while not finished_event.is_set() or not queue_out.empty():
                try:
                    audio_chunk = queue_out.get_nowait()
                    
                    audio_b64 = self.audio_to_base64_pcm(audio_chunk)
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "audio_data": audio_b64
                    })
                    chunk_count += 1
                    await asyncio.sleep(0.001)
                    
                except queue.Empty:
                    await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"❌ Error streaming audio: {e}", exc_info=True)

        finally:
            await websocket.send_json({"type": "audio_stream_end"})
            logger.info(f"✅ Streamed {chunk_count} audio chunks")


# --- Helper Functions ---
async def clean_markdown(text: str) -> str:
    """Remove markdown formatting from text."""
    return re.sub(r"\*\*|#{1,6}\s*|- ", '', text, flags=re.MULTILINE).strip()


async def consume_websocket_messages(websocket: WebSocket):
    """
    Non-blocking consumption of websocket messages to prevent queue buildup and timeouts.
    Useful during phases where we are generating silence but client might still be sending data.
    """
    try:
        while True:
            try:
                # Use a very short timeout to check for messages
                msg = await asyncio.wait_for(websocket.receive(), timeout=0.001)
                
                if msg.get("type") == "websocket.disconnect":
                    logger.info("🔌 Client initiated disconnect during silence phase")
                    raise WebSocketDisconnect
                
                # Check for ping
                if 'text' in msg:
                    try:
                        text_msg = json.loads(msg['text'])
                        if text_msg.get('type') == 'ping':
                            await websocket.send_json({"type": "pong"})
                    except:
                        pass
                
                # We basically discard audio input during silence generation phases 
                # (prefix/suffix) because we are simulated/flushing.
                # Just keeping the socket alive.
                
            except asyncio.TimeoutError:
                # No more messages immediately available
                break
    except WebSocketDisconnect:
        raise
    except Exception as e:
        logger.warning(f"⚠️ Error in non-blocking consume: {e}")


async def call_lead_creation_api(messages: list, session_id: str,
                                 tts_handler: TTSStreamingHandler,
                                 websocket: WebSocket) -> str:
    """Call LLM API and stream both text and audio responses."""
    final_assistant_response = ""
    accumulated_text = ""

    logger.info(f"🌐 Calling lead creation API with {len(messages)} messages")

    try:
        history = [{"sender": msg["role"], "content": msg["content"]} for msg in messages]
        payload = {"history": history}

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            async with session.post(
                    "http://zcrmdit-a6k-1:8090/text/api/llm/agents/trigger_agent_orchestration",
                    json=payload
            ) as response:
                response.raise_for_status()

                async for chunk in response.content:
                    if not chunk:
                        continue

                    try:
                        for line in chunk.decode('utf-8').splitlines():
                            if not line.strip():
                                continue

                            try:
                                current_chunk_text = json.loads(line.strip()).get('data', '')
                                if not current_chunk_text:
                                    continue

                                final_assistant_response += current_chunk_text
                                accumulated_text += current_chunk_text

                                # When we have enough text (sentence or clause), generate TTS
                                if any(punct in accumulated_text for punct in ['.', '!', '?', ',', ';']):
                                    cleaned_text = await clean_markdown(accumulated_text)
                                    if cleaned_text.strip():
                                        await tts_handler.generate_and_stream(
                                            websocket,
                                            cleaned_text.strip()
                                        )
                                    accumulated_text = ""

                            except json.JSONDecodeError:
                                continue
                    except UnicodeDecodeError:
                        continue

                # Generate TTS for any remaining text
                if accumulated_text.strip():
                    cleaned_text = await clean_markdown(accumulated_text)
                    if cleaned_text.strip():
                        await tts_handler.generate_and_stream(websocket, cleaned_text.strip())

                logger.info(f"✅ API call completed. Response: {len(final_assistant_response)} chars")

    except Exception as e:
        logger.error(f"❌ Error in lead creation API: {e}", exc_info=True)
        final_assistant_response = "I'm sorry, I encountered an error. Please try again."
        await tts_handler.generate_and_stream(websocket, final_assistant_response)

    return final_assistant_response


@app.websocket("/audio/api/leadcreation/agent/create")
async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"🔌 Client connected from {websocket.client.host}:{websocket.client.port}")

    session_id = None
    input_sample_rate = 24000
    
    try:
        # Wait for initial configuration from client
        init_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        logger.info(f"📥 Received initial config: {init_msg}")
        session_id = init_msg.get("session_id") or str(uuid.uuid4())
        input_sample_rate = init_msg.get("sampleRate", 24000)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        logger.warning("⚠️ Client did not send config. Assigning new session ID.")
        session_id = str(uuid.uuid4())
    except Exception as e:
        logger.error(f"❌ Error during handshake: {e}", exc_info=True)
        await websocket.close(code=1011, reason="Handshake error")
        return

    # Load session BEFORE heavy initialization
    session = load_session(session_id)
    
    # Send session ready confirmation FIRST
    await websocket.send_json({
        "type": "session_ready",
        "session_id": session_id,
        "stt_sample_rate": model_manager.stt_mimi.sample_rate,
        "tts_sample_rate": model_manager.tts_sample_rate
    })
    
    logger.info(f"✅ Session ready message sent: {session_id}")

    # NOW initialize STT state (this is heavy and takes time)
    logger.info("🔧 Initializing STT state...")
    stt_state = InferenceState(
        model_manager.stt_mimi,
        model_manager.stt_text_tokenizer,
        model_manager.stt_lm,
        batch_size=1,
        device=model_manager.device,
        stt_config=model_manager.stt_checkpoint_info.stt_config or {},
        raw_config=model_manager.stt_checkpoint_info.raw_config or {}
    )
    tts_handler = TTSStreamingHandler(model_manager.tts_model)
    logger.info("✅ STT state initialized")

    # --- VAD Configuration ---
    silence_chunk = torch.zeros((1, 1, stt_state.frame_size), dtype=torch.float32, device=stt_state.device)
    n_suffix_chunks = math.ceil(stt_state.audio_delay_seconds * stt_state.mimi.frame_rate)
    vad_threshold = 0.5
    last_print_was_vad = False

    # Flag to track if we need to send greeting
    need_greeting = not session["messages"]

    try:
        # Main conversation loop - THIS MUST RUN CONTINUOUSLY
        while True:
            sample_buffer = []
            vad_triggered = False
            suffix_chunks_fed = 0
            text_tokens_accum = []

            # Send greeting at the START of the first turn (inside the loop!)
            if need_greeting:
                logger.info("📢 Sending initial greeting")
                await websocket.send_json({"type": "audio_stream_start"})
                greeting_text = "Hello! I'm ready to help. Please go ahead and speak."
                session["messages"].append({"role": "assistant", "content": greeting_text})
                await tts_handler.generate_and_stream(websocket, greeting_text)
                save_session(session_id, session)
                need_greeting = False
                logger.info("✅ Initial greeting sent, now listening for user input...")

            logger.info("🎤 Listening for user speech...")

            # Reset STT state for new turn
            stt_state.reset()

            # Use streaming context manager
            with stt_state.mimi.streaming(stt_state.batch_size), stt_state.lm_gen.streaming(stt_state.batch_size):
                utterance_complete = False

                while not utterance_complete:
                    codes = None

                    # PHASE 1: Feed silence prefix for VAD calibration
                    if stt_state.is_in_prefix_phase:
                        if stt_state.prefix_chunks_fed < stt_state.n_prefix_chunks:
                            codes = stt_state.mimi.encode(silence_chunk)
                            stt_state.prefix_chunks_fed += 1
                            
                            if stt_state.prefix_chunks_fed == 1 or stt_state.prefix_chunks_fed % 10 == 0:
                                logger.debug(f"🔇 Feeding prefix silence: {stt_state.prefix_chunks_fed}/{stt_state.n_prefix_chunks}")
                            
                            # CRITICAL: Drain websocket during prefix phase to keep connection alive
                            await consume_websocket_messages(websocket)

                        else:
                            stt_state.is_in_prefix_phase = False
                            logger.info(f"✅ Silence prefix complete. Now actively listening for audio input.")
                            continue
                    
                    # PHASE 2: VAD triggered, feed suffix silence
                    elif vad_triggered:
                        if suffix_chunks_fed < n_suffix_chunks:
                            codes = stt_state.mimi.encode(silence_chunk)
                            suffix_chunks_fed += 1
                            
                            if suffix_chunks_fed % 20 == 0:
                                logger.debug(f"🔇 Feeding suffix silence: {suffix_chunks_fed}/{n_suffix_chunks}")
                            
                            # CRITICAL: Drain websocket during suffix phase to keep connection alive
                            await consume_websocket_messages(websocket)

                        else:
                            # Utterance complete
                            final_text = stt_state.accumulated_text.strip()
                            logger.info(f"✅ Turn complete. Transcription: '{final_text}'")

                            await websocket.send_json({"type": "final_transcription", "text": final_text})
                            
                            if final_text:
                                # Send audio stream start signal
                                await websocket.send_json({"type": "audio_stream_start"})

                                # Process with LLM
                                try:
                                    session["messages"].append({"role": "user", "content": final_text})
                                    assistant_response = await call_lead_creation_api(
                                        session["messages"], session_id, tts_handler, websocket
                                    )
                                    session["messages"].append({"role": "assistant", "content": assistant_response})
                                    save_session(session_id, session)
                                    logger.info("✅ Turn processed successfully")
                                except Exception as e:
                                    logger.error(f"❌ Error processing turn: {e}", exc_info=True)
                            else:
                                logger.info("⚠️ Empty transcription, skipping LLM call")

                            utterance_complete = True
                            last_print_was_vad = False
                            # CRITICAL: Break out of inner loop to continue outer loop
                            break
                    
                    # PHASE 3: Normal listening - receive audio from client
                    else:
                        try:
                            # Set a reasonable timeout to detect disconnections
                            msg = await asyncio.wait_for(websocket.receive(), timeout=60.0)
                            
                            if msg.get("type") == "websocket.disconnect":
                                logger.info("🔌 Client initiated disconnect")
                                raise WebSocketDisconnect
                            
                            # Handle text messages (control signals)
                            if 'text' in msg:
                                try:
                                    text_msg = json.loads(msg['text'])
                                    msg_type = text_msg.get('type')
                                    
                                    if msg_type == 'pong':
                                        logger.debug("📡 Received pong from client")
                                    elif msg_type == 'vad_start':
                                        logger.info("🟢 Client signaled VAD START")
                                        # Optionally reset state if needed, but usually just keep streaming
                                        pass
                                    elif msg_type == 'vad_stop':
                                        logger.info("🔴 Client signaled VAD STOP")
                                        vad_triggered = True
                                        # We don't need to feed suffix silence here necessarily if we just want to force-stop
                                        # But keeping consistent with the logic:
                                        # Break the loop effectively by setting the flag
                                except:
                                    pass
                                continue

                            elif 'bytes' in msg:
                                data = msg['bytes']
                                try:
                                    samples = (np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0).tolist()
                                except Exception:
                                    samples = np.frombuffer(data, dtype=np.float32).tolist()

                                sample_buffer.extend(samples)
                                input_frame_size = int(
                                    stt_state.frame_size * (input_sample_rate / stt_state.mimi.sample_rate) + 0.5)

                                if len(sample_buffer) >= input_frame_size:
                                    chunk_samples = np.array(sample_buffer[:input_frame_size], dtype=np.float32)
                                    sample_buffer = sample_buffer[input_frame_size:]

                                    if int(input_sample_rate) != int(stt_state.mimi.sample_rate):
                                        chunk_samples = signal.resample(chunk_samples, stt_state.frame_size)

                                    chunk = torch.from_numpy(chunk_samples).to(stt_state.device).unsqueeze(0).unsqueeze(0)
                                    codes = stt_state.mimi.encode(chunk)
                            else:
                                continue
                                
                        except asyncio.TimeoutError:
                            logger.warning("⏱️ No audio received for 60s")
                            # Send a ping to check if connection is alive
                            try:
                                await websocket.send_json({"type": "ping"})
                            except:
                                logger.error("❌ Failed to ping client, connection lost")
                                raise WebSocketDisconnect
                            continue

                    if codes is None:
                        continue

                    # Process with STT model
                    text_tokens, vad_heads = stt_state.lm_gen.step_with_extra_heads(codes)
                    text_tokens_accum.append(text_tokens)

                    # Process text tokens
                    if text_tokens is not None:
                        text_token = int(text_tokens[0, 0, 0].cpu().item())
                        
                        if text_token not in (0, stt_state.padding_token_id):
                            stt_state.accumulated_token_ids.append(text_token)
                            
                            decoded_so_far = stt_state.text_tokenizer.decode(stt_state.accumulated_token_ids)
                            new_text = decoded_so_far[len(stt_state.accumulated_text):]

                            if new_text.strip():
                                await websocket.send_json({"type": "partial_transcription", "text": new_text})
                                stt_state.accumulated_text = decoded_so_far
                                last_print_was_vad = False

                    # SERVER-SIDE VAD CHECK - DISABLED (LOGGING ONLY)
                    if vad_heads and not stt_state.is_in_prefix_phase:
                        pr_vad = vad_heads[2][0, 0, 0].cpu().item()
                        if pr_vad > vad_threshold and not last_print_was_vad:
                            logger.debug(f"👀 Server VAD detected likely silence (pr_vad={pr_vad:.3f}) - ignoring in favor of client VAD")
                            last_print_was_vad = True

            # After utterance is complete, loop continues to next turn
            logger.info("🔄 Turn complete, ready for next user input")

    except WebSocketDisconnect:
        logger.info(f"🔌 Client disconnected from session {session_id}")
    except Exception as e:
        if "CUDA" in str(e):
            logger.error(f"❌ CUDA error: {e}", exc_info=True)
            torch.cuda.empty_cache()
        else:
            logger.error(f"❌ Unhandled error: {e}", exc_info=True)
    finally:
        logger.info(f"🏁 Closing WebSocket for session {session_id}")
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting audio server...")
    uvicorn.run(
        app, host="0.0.0.0", port=8003,
        ssl_certfile="server.crt", ssl_keyfile="server.key", log_level="info"
    )
