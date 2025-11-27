from fastapi import FastAPI, UploadFile
import tempfile
from vocalid.verifier import VoiceVerifier

verifier = VoiceVerifier("voice_auth.pkl")
app = FastAPI()

@app.post("/verify")
async def verify_voice(file: UploadFile):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(await file.read())
    tmp.close()
    ok, score = verifier.verify_file(tmp.name)
    return {"verified": ok, "score": score}
       