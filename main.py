import aiohttp_cors
from aiohttp import web
import aiohttp
import os
import sys
import subprocess
import time
import requests
import asyncio
from multiprocessing import Process
from loguru import logger

from pipecat.vad.vad_analyzer import VADParams
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.services.openai import OpenAILLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.pipeline import Pipeline
from pipecat.frames.frames import LLMMessagesFrame, EndFrame

from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator
)
from model import load_model
from helpers import (
    ClearableDeepgramTTSService,
    AudioVolumeTimer,
    TranscriptionTimingLogger
)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

os.environ['SSL_CERT'] = ''
os.environ['SSL_KEY'] = ''
os.environ['OUTLINES_CACHE_DIR'] = '/tmp/.outlines'

deepgram_voice: str = "aura-asteria-en"

# Manually set API keys
openai_api_key = "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"
daily_api_key = "9929b1fef86091d59f4524358f970bc47328f17501d8fdf5052b6a9a9b046d77"


model_id = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
model = load_model(model_id)

# Run vllM Server in background process
def start_server():
    while True:
        process = subprocess.Popen(
            f"python -m vllm.entrypoints.openai.api_server --port 5000 --model {model} --api-key {openai_api_key}",
            shell=True
        )
        process.wait()  # Wait for the process to complete
        logger.error("Server process ended unexpectedly. Restarting in 5 seconds...")
        time.sleep(7)  # Wait before restarting

# Start the server in a separate process
server_process = Process(target=start_server, daemon=True)
server_process.start()

async def main(room_url: str, token: str):
    
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(
                    stop_secs=0.2
                )),
                vad_audio_passthrough=True
            )
        )

        stt = DeepgramSTTService(
            name="STT",
            api_key=None,
            url='ws://127.0.0.1:8082/v1/listen'
        )

        tts = ClearableDeepgramTTSService(
            name="Voice",
            aiohttp_session=session,
            api_key=None,
            voice=deepgram_voice,
            base_url="http://127.0.0.1:8082/v1/speak"
        )
        
        llm = OpenAILLMService(
            name="LLM",
            api_key=openai_api_key,
            model=model_id,
            base_url="http://127.0.0.1:5000/v1"  # Ensure this matches your server's URL
        )

        messages = [
            {
                "role": "system",
                "content": "You are a fast, low-latency chatbot. Your goal is to demonstrate voice-driven AI capabilities at human-like speeds. The technology powering you is Daily for transport, Cerebrium for serverless infrastructure, Llama 3 (8-B version) LLM, and Deepgram for speech-to-text and text-to-speech. You are hosted on the east coast of the United States. Respond to what the user said in a creative and helpful way, but keep responses short and legible. Ensure responses contain only words. Check again that you have not included special characters other than '?' or '!'.",
            },
        ]

        avt = AudioVolumeTimer()
        tl = TranscriptionTimingLogger(avt)

        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator(messages)

        pipeline = Pipeline([
            transport.input(),   # Transport user input
            avt,                 # Audio volume timer
            stt,                 # Speech-to-text
            tl,                  # Transcription timing logger
            tma_in,              # User responses
            llm,                 # LLM
            tts,                 # TTS
            transport.output(),  # Transport bot output
            tma_out,             # Assistant spoken responses
        ])

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                report_only_initial_ttfb=True
            ))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            # Kick off the conversation.
            time.sleep(1.5)
            messages.append(
                {"role": "system", "content": "Introduce yourself by saying 'hello, I'm FastBot, how can I help you today?'"}
            )
            await task.queue_frame(LLMMessagesFrame(messages))

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.queue_frame(EndFrame())

        @transport.event_handler("on_call_state_updated")
        async def on_call_state_updated(transport, state):
            if state == "left":
                await task.queue_frame(EndFrame())

        runner = PipelineRunner()

        await runner.run(task)
        await session.close()

def check_vllm_model_status():
    url = "http://127.0.0.1:5000/v1/models"
    headers = {
        "Authorization": f"Bearer {openai_api_key}"
    }
    max_retries = 8
    for _ in range(max_retries):
        print('Trying vllm server')
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            print("Connection refused, retrying...")
        time.sleep(15)
    return False

def start_bot(room_url: str, token: str = None):

    def target():
        asyncio.run(main(room_url, token))

    check_vllm_model_status()
    process = Process(target=target, daemon=True)
    process.start()
    process.join()  # Wait for the process to complete
    return {"message": "session finished"}

def create_room():
    url = "https://api.daily.co/v1/rooms/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {daily_api_key}"
    }
    data = {
        "properties": {
            "exp": int(time.time()) + 60*5, ##5 mins
            "eject_at_room_exp" : True
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        room_info = response.json()
        token = create_token(room_info['name'])
        if token and 'token' in token:
            room_info['token'] = token['token']
        else:
            logger.error("Failed to create token")
            return {"message": 'There was an error creating your room', "status_code": 500}
        return room_info
    else:
        data = response.json()
        if data.get("error") == "invalid-request-error" and "rooms reached" in data.get("info", ""):
            logger.error("We are currently at capacity for this demo. Please try again later.")
            return {"message": "We are currently at capacity for this demo. Please try again later.", "status_code": 429}
        logger.error(f"Failed to create room: {response.status_code}")
        return {"message": 'There was an error creating your room', "status_code": 500}

def create_token(room_name: str):
    url = "https://api.daily.co/v1/meeting-tokens"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {daily_api_key}"
    }
    data = {
        "properties": {
            "room_name": room_name
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        token_info = response.json()
        return token_info
    else:
        logger.error(f"Failed to create token: {response.status_code}")
        return None

async def handle_create_room(request):
    room_info = create_room()
    return web.json_response(room_info)

async def handle_start_bot(request):
    request_data = await request.json()
    room_url = request_data.get('room_url')
    token = request_data.get('token')
    start_bot(room_url, token)
    return web.json_response({"message": "bot started"})

app = web.Application()

# Add your routes here
app.router.add_post('/v1/voice_agent', handle_start_bot)

# Configure default CORS settings.
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*"
    )
})

# Configure CORS on all routes.
for route in list(app.router.routes()):
    cors.add(route)

# Run the application
web.run_app(app, port=3000)
