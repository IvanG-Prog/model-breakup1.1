import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

# Load environment variables (TOKEN and CHAT_ID must be set in the root .env)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

app = FastAPI(title="Telegram Proxy Service")

class AlertPayload(BaseModel):
    message: str

@app.post("/send_alert")
async def proxy_telegram_alert(payload: AlertPayload):
    """Receives alert message and forwards it to the Telegram API."""
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        # This error means the environment setup in Docker Compose is wrong
        return {"status": "error", "detail": "Proxy service missing Telegram credentials."}

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    tg_payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': payload.message,
        'parse_mode': 'HTML'
    }
    
    try:
        # Forward the request to Telegram API
        response = requests.post(url, data=tg_payload, timeout=10)
        response.raise_for_status()
        
        print(f"✅ Proxy: Alert forwarded successfully. Status: {response.status_code}")
        return {"status": "success", "detail": "Message proxied successfully."}
        
    except requests.exceptions.RequestException as e:
        error_detail = f"Failed to send to Telegram from Proxy: {type(e).__name__} - {e}"
        print(f"❌ Proxy ERROR: {error_detail}")
        return {"status": "error", "detail": error_detail}

# --- NUEVO ENDPOINT PARA KEEP-ALIVE ---
@app.get("/keep_alive")
async def keep_alive():
    """Endpoint de ping para evitar que el proxy entre en modo de suspensión."""
    print("🟢 Proxy: Ping keep-alive recibido.")
    return {"status": "success", "detail": "Proxy está activo y responde."}
