# ai-assistant
# AI Resume ATS Checker

Upload a resume (PDF, DOCX or TXT) and get:

- An overall **ATS score** (0-100) plus a breakdown by category
- Strengths and **missing keywords**
- Prioritized **improvements** with exact fixes
- Before/after **rewrite examples**
- Optional **job description** matching for better keyword analysis

Built with [Streamlit](https://streamlit.io) and Google's Gemini Flash model.

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"   # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

Get a free API key at https://aistudio.google.com/apikey. You can also paste the key
into the app's sidebar instead of setting the environment variable.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (`app.py`, `requirements.txt`, `README.md`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings -> Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Configuration

- The default model is set by `DEFAULT_MODEL` in `app.py` and can be changed in the sidebar
  (for example `gemini-3.5-flash`, or an older Flash model if your key does not have access).
- Never commit your API key to GitHub.

## Limitations

- Scanned/image-only PDFs have no extractable text; export a text-based PDF or DOCX instead.
- The score is an AI estimate, not the output of a real employer ATS.
- Resume text is sent to the Gemini API for analysis. Do not upload anything you are not comfortable sharing.
