# Text to 3D AI Model

Simple FastAPI web app that generates a source image from a text prompt, converts it to a 3D mesh with TripoSR, previews the GLB in the browser, and lets the user download the model.

## Local Setup

```powershell
cd C:\Users\ISMS\Downloads\practice
.\venv\Scripts\python.exe api.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Git Commit And Push

Check what changed:

```powershell
git status
git diff --stat
```

Commit and push to the existing GitHub repository:

```powershell
git add api.py index.html pipeline.py requirements.txt TripoSR/requirements.txt README.md Text_to_3D_Colab.ipynb
git commit -m "Improve text to 3D generation and Colab hosting"
git push origin main
```

The configured remote is:

```text
https://github.com/LucyAlex12/Text_to_3d_ai_model.git
```

## Google Colab Public Hosting

1. Open `Text_to_3D_Colab.ipynb` in Google Colab.
2. In Colab, use `Runtime > Change runtime type > T4 GPU`.
3. Add your ngrok authtoken in Colab secrets as `NGROK_AUTH_TOKEN`, or paste it when the notebook asks.
4. Optional: reserve a static domain in ngrok and add it to Colab secrets as `NGROK_STATIC_DOMAIN`, for example `your-name.ngrok-free.app`.
5. Run the notebook cells from top to bottom.
6. Open the printed ngrok URL in a normal browser tab, not Colab's iframe preview.

If you do not set `NGROK_STATIC_DOMAIN`, ngrok gives a temporary URL that changes each time the notebook restarts. If you do set a static domain, the URL can stay the same, but the app still only works while the Colab runtime is running.

## Notes

- `TripoSR/model.ckpt` is large and should not be committed to Git. The Colab notebook downloads it from Hugging Face.
- Hugging Face may show an unauthenticated-download warning in Colab. It is safe to ignore for the public TripoSR files unless rate limits interrupt the download.
- `shap-e` is not required by this app and is intentionally not installed because it does not support Colab's current Python 3.12 runtime.
- The frontend uses same-origin API paths, so it works locally and through ngrok.
- The frontend sends the `ngrok-skip-browser-warning` header for API, model-preview, and download requests.
- The download button fetches the generated `.glb` file and saves it as `text-to-3d-model.glb`.
