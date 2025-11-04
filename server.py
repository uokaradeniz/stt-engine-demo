import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.concurrency import run_in_threadpool
from typing import Optional

from transcription import process_audio

app = FastAPI()


@app.post("/transcribe")
async def transcribe(request: Request, file: Optional[UploadFile] = File(None), audio: Optional[UploadFile] = File(None)):
    upload: Optional[UploadFile] = file or audio

    if upload is None:
        # Try to find any UploadFile in the form (handles other field names)
        form = await request.form()
        for value in form.values():
            if isinstance(value, UploadFile):
                upload = value
                break

    if upload is None:
        # Return a 400 with a helpful message instead of letting FastAPI return 422
        raise HTTPException(status_code=400, detail="No file uploaded. Use form field 'audio' or 'file'.")

    contents = await upload.read()

    # Run the blocking transcription in a threadpool
    result = await run_in_threadpool(lambda: process_audio(contents))
    return result["text"]


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
