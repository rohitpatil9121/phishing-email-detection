# Deploying on Render

This is a single Render **web service** that serves the backend *and* the frontend —
FastAPI mounts `static/` and returns `index.html` at `/`, so there is nothing else to host.

`model.pkl` (~220 KB) is committed, so Render does **not** retrain at build time. The build
is just `pip install`, which stays inside the free tier's memory and time limits.

## 1. Push to GitHub

```bash
git init
git add .
git commit -m "Phishing email detection web app"
git branch -M main
git remote add origin https://github.com/rohitpatil9121/phishing-email-detection.git
git push -u origin main
```

`.gitignore` excludes `.env` and `data/dataset.csv` (~48 MB). Anyone cloning runs
`python download_data.py && python train.py` to rebuild it.

## 2. Create the service

Render → **New** → **Blueprint** → select this repo. It reads [render.yaml](render.yaml)
and creates `mailguard-phishing` at `https://mailguard-phishing.onrender.com`.

Manual alternative — New → Web Service, then:

- Runtime **Python 3**
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## 3. Add the Groq key

Service → **Environment** → add `GROQ_API_KEY` → save (this redeploys). `llm.py` reads the
environment variable first, so no `.env` is needed in production. Leave it unset and the
site still works — the Groq radios disable themselves and the trained model handles
everything.

## 4. Check it

```
https://<service>.onrender.com/            # the site
https://<service>.onrender.com/docs        # interactive API
https://<service>.onrender.com/api/info    # model + LLM status
```

`/api/info` is the blueprint's health check, so a failed model load fails the deploy
instead of silently serving a broken page.

## Free-tier caveats

- Services **sleep after 15 minutes** idle; the next request takes ~30–50 s to wake. Open
  the URL a minute before demoing it.
- 512 MB RAM — fine for a 220 KB model plus scikit-learn/pandas, but do not add
  `download_data.py` / `train.py` to the build command.
- To retrain: run `train.py` locally, commit the new `model.pkl`, push. Render redeploys
  automatically.
