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

# --- ENDPOINT DE SALUD PARA LA RUTA RAÍZ ---
@app.get("/")
async def root_health_check():
    """Endpoint de salud simple para la ruta raíz (/)."""
    return {"status": "ok", "service": "Telegram Proxy"}

@app.post("/send_alert")
async def proxy_telegram_alert(payload: AlertPayload):
    """Receives alert message and forwards it to the Telegram API."""
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return {"status": "error", "detail": "Proxy service missing Telegram credentials."}

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Intento 1: con parse_mode='HTML'
    tg_payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': payload.message,
        'parse_mode': 'HTML'
    }
    
    try:
        # Intento de envío a Telegram
        response = requests.post(url, data=tg_payload, timeout=10)
        response.raise_for_status()
        
        print(f"✅ Proxy: Alert forwarded successfully. Status: {response.status_code}")
        return {"status": "success", "detail": "Message proxied successfully."}
        
    except requests.exceptions.HTTPError as e:
        # Si el error es 400 (Bad Request), es probablemente un problema de parseo HTML
        if e.response.status_code == 400:
            print(f"⚠️ Proxy WARNING: Primer intento falló con 400 Bad Request. Contenido HTML no válido. Reintentando sin parse_mode.")
            
            # Intento 2: sin parse_mode (texto plano)
            tg_payload.pop('parse_mode', None)
            
            try:
                # Reintento de envío a Telegram en modo texto plano
                response = requests.post(url, data=tg_payload, timeout=10)
                response.raise_for_status()
                
                print(f"✅ Proxy: Alert forwarded successfully on 2nd attempt (Plain Text).")
                return {"status": "success", "detail": "Message proxied successfully (Plain Text fallback)."}

            except requests.exceptions.RequestException as retry_e:
                # Si falla el reintento, loguear el error final
                error_detail = f"Failed to send to Telegram from Proxy after retry: {type(retry_e).__name__} - {retry_e}"
                print(f"❌ Proxy ERROR: {error_detail}")
                return {"status": "error", "detail": error_detail}
        
        else:
            # Si es otro error HTTP (401, 500, etc.), loguear el error original
            error_detail = f"Failed to send to Telegram from Proxy: {type(e).__name__} - {e}"
            print(f"❌ Proxy ERROR: {error_detail}")
            return {"status": "error", "detail": error_detail}
            
    except requests.exceptions.RequestException as e:
        # Manejo de errores de conexión/timeout que no son HTTPError
        error_detail = f"Failed to send to Telegram from Proxy (Connection Error): {type(e).__name__} - {e}"
        print(f"❌ Proxy ERROR: {error_detail}")
        return {"status": "error", "detail": error_detail}

# --- ENDPOINT PARA KEEP-ALIVE ---
@app.get("/keep_alive")
async def keep_alive():
    """Endpoint de ping para evitar que el proxy entre en modo de suspensión."""
    print("🟢 Proxy: Ping keep-alive recibido.")
    return {"status": "success", "detail": "Proxy está activo y responde."}
