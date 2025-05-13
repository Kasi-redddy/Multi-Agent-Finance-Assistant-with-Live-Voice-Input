# Multi-Agent Finance Assistant with Live Voice Input

A Streamlit-based AI assistant for financial portfolio managers and analysts.  
Ask questions about your portfolio or the market using **live voice recording** or text, and receive concise, AI-generated market briefs with text-to-speech playback.

---

## Features

- 🎙️ **Live voice input**: Record your question directly in the browser (no file upload needed).
- 📝 **Text input**: Type your question if you prefer.
- 🗣️ **Speech-to-text**: Uses OpenAI Whisper for accurate transcription.
- 🧑‍💼 **Multi-agent data retrieval**:
  - Fetches latest stock data from Yahoo Finance.
  - Scrapes financial metrics from the web.
  - Retrieves relevant context from a FAISS vector store.
  - Performs portfolio exposure analysis.
- 🤖 **AI-generated market briefs**: Summarizes all data into a concise, actionable brief.
- 🔊 **Text-to-speech**: Play the answer aloud in your browser.
- 🚀 **FastAPI backend**: Modular API for future extensibility.

---

## Demo

<!-- Add a screenshot or GIF here if you like -->
![Demo Screenshot](demo_screenshot.png)

---

## Installation

1. **Clone the repository:**
    ```
    git clone https://github.com/yourusername/multi-agent-finance-assistant.git
    cd multi-agent-finance-assistant
    ```

2. **(Optional) Create and activate a virtual environment:**
    ```
    conda create -n finance_assistant_env python=3.9 -y
    conda activate finance_assistant_env
    ```

3. **Install Python dependencies:**
    ```
    pip install -r requirements.txt
    ```

4. **Install ffmpeg (required for Whisper):**
    - With conda:
      ```
      conda install -c conda-forge ffmpeg
      ```
    - Or [download from ffmpeg.org](https://ffmpeg.org/download.html) and add to your system PATH.

---

## Usage

1. **Start the app:**
    ```
    streamlit run app.py
    ```

2. **Open your browser at** [http://localhost:8501](http://localhost:8501).

3. **Ask your question:**
    - Click the 🎙️ button to record your question (press again to stop).
    - Or type your question in the text box.

4. **Get your answer:**
    - Click **Get Market Brief**.
    - Read the AI-generated summary or click **🔊 Play Voice** to hear it.

---

## Project Structure

- `app.py` - Main Streamlit and FastAPI app.
- `requirements.txt` - Python dependencies.
- `README.md` - Project documentation.

---

## Notes

- The default language model is `distilgpt2` (small and fast). For better summaries, swap in a more powerful model (e.g., OpenAI GPT-3.5-turbo or Mistral).
- Whisper model weights are downloaded on first run; ensure a stable internet connection.
- `ffmpeg` must be installed and accessible in your system PATH for audio processing.

---

## Contact

For questions or contributions, please open an issue or submit a pull request.

---




