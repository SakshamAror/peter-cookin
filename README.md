# Peter Cookin

Your camera-wielding kitchen sidekick: spot ingredients, get recipe ideas, and ask for cooking guidance without leaving the camera view.

Peter Cookin is a local macOS desktop app built with Python and Tkinter. It uses OpenCV and Ultralytics YOLOE with ByteTrack to propose and track food and pantry-item labels. Gemini reviews uncertain items, reads visible package text, suggests recipes, and checks cooking progress from camera snapshots. Voiskey can dictate into the prompt field, and ElevenLabs reads Gemini's replies aloud. Apple Vision OCR helps read English package labels locally.

## Run it

1. Install Python 3.12 or newer and Swift (included with Xcode Command Line Tools).
2. In Terminal, go to this folder and start the app:

   ```sh
   ./run.sh
   ```

On first launch, the script creates `.venv`, installs the requirements, and downloads YOLOE model files. Allow Terminal or Python to access the camera when macOS asks.

## API keys

Copy `.env.example` to `.env`, then add your Gemini and ElevenLabs credentials. The `.env` file is ignored by Git. Keep API keys out of source files and commits. Voiskey works by dictating into the prompt field; press Return to submit.

## How it works

- Local YOLOE proposals are tracked as items move; labels appear after Gemini confirms food or cooking ingredients.
- The first Gemini label review starts two seconds after launch. Further reviews run every five seconds for new candidates.
- Recipe and cooking-progress prompts include a camera snapshot and recent item labels. The continuous camera feed stays local.
- Correct or remove labels in the prompt field; these simple edits are handled locally.
- Gemini replies appear in the sidebar and are spoken through the configured ElevenLabs voice.

Gemini's visual judgments and nutrition estimates are suggestions. A camera image cannot confirm internal food temperature or guarantee food safety; use a food thermometer when appropriate.

## Built with

Python, Tkinter, OpenCV, Ultralytics YOLOE, ByteTrack, Google Gemini API, Voiskey, ElevenLabs Text-to-Speech, and Apple Vision OCR. Google AI Studio was used during development.
