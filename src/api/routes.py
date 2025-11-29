import asyncio 
import os
import httpx 
from fastapi import APIRouter, Request, HTTPException

# Import necessary core functions
# NOTE: The background_scheduler may need run_scanner_main if it is executing the full scan.
from src.scheduler.background_scheduler import run_async_scanner 
from src.model_pipeline.scanner_live import send_ping_message 
from src.model_pipeline.scanner_live import run_single_symbol_prediction

# --- FastAPI Router Definition (CRITICAL FOR IMPORT) ---
# This is the variable the main app file (app.py) expects to import.
router = APIRouter()

# --- Utility Endpoint ---
@router.get("/utility")
async def utility_endpoint(action: str):
    """
    Utility endpoint to manually trigger scans or connection tests.
    
    Args:
        action (str): The desired action ('test_connection' or 'run_scan').
    """
    if action == 'test_connection':
        test_message = "🛠️ **MANUAL CONNECTION TEST**\nSuccessful connectivity check via API endpoint."
        
        try:
            # We use the existing channel ping function here
            success = send_ping_message(test_message)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"❌ Test alert sending failed due to exception: {e}")

        if success:
            return {"status": "success", "message": "✅ Test alert successfully sent via configured Telegram method."}
        else:
            return {"status": "error", "message": "❌ Test alert sending failed. Check environment variables and server logs."}
            
    elif action == 'run_scan':
        # Execute the scanner, source is MANUAL_API
        # We need to run this synchronously because it's an API request blocking the thread
        alert_sent = run_async_scanner(source='MANUAL_API')

        if alert_sent is True:
            return {"status": "success", "message": "✅ Strong signal found! Alert sent to Telegram."}
        elif alert_sent is False:
            return {"status": "info", "message": "🟡 Prediction found, but Net Advantage was below the 70.0% alert threshold. No message sent."}
        else:
            raise HTTPException(status_code=500, detail="Failed to run scanner due to a threading/async execution error.")
            
    else:
        raise HTTPException(status_code=400, detail="Invalid action specified. Use 'test_connection' or 'run_scan'.")


# --- Telegram Reply Utility ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

async def send_telegram_reply(chat_id: str, message: str):
    """
    Sends a message response back to a specific Telegram user/chat ID.
    """
    if not TELEGRAM_BOT_TOKEN:
        print("--- ⚠️ TELEGRAM_BOT_TOKEN is not set. Cannot send user reply. ---")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        # Use httpx for asynchronous API calls
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=10)
            response.raise_for_status() 
            return True
    except Exception as e:
        print(f"--- ❌ Error sending Telegram reply: {e} ---")
        return False


# --- Telegram Webhook Endpoint ---
@router.post("/telegram_webhook")
async def telegram_webhook(request: Request):
    """
    Receives updates from the Telegram Bot API. 
    Handles user commands: /scan, /analyze [symbol], and /features.
    """
    chat_id = None
    
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON from Telegram.")

    try:
        message = data.get('message')
        if not message:
            return {"status": "ok", "message": "No message in update."}
            
        chat_id = str(message['chat']['id'])
        message_text = message.get('text', '').strip() 
        lower_message_text = message_text.lower()
        
        # --------------------------------------------------------------------------------
        # 1. /SCAN COMMAND (Triggers full scheduled scan logic - alerts to channel)
        # --------------------------------------------------------------------------------
        if lower_message_text == '/scan':
            print(f"--- 🤖 TELEGRAM COMMAND RECEIVED from {chat_id}: /scan. Triggering immediate scan... ---")
            
            # El primer mensaje debe enviarse antes de la función de bloqueo
            await send_telegram_reply(chat_id, "🔍 Initiating full market scan. Please wait (this can take 30-60 seconds)...")

            # run_async_scanner debe estar diseñado para ejecutarse en un threadpool
            alert_found = run_async_scanner(source='TELEGRAM_WEBHOOK')
            
            if alert_found is True:
                response_text = "✅ Scan completed! A strong signal has been detected and sent to the alert channel."
            elif alert_found is False:
                response_text = "🟡 Scan completed. No strong signal (Net Advantage < 70%) found at this time."
            else:
                response_text = "❌ Scan failed due to an internal execution error. Check server logs."
                
            await send_telegram_reply(chat_id, response_text)
            return {"status": "ok", "message": "Scan command processed."}

        # --------------------------------------------------------------------------------
        # 2. /ANALYZE [SYMBOL] COMMAND (Triggers REAL PREDICTION for a single symbol)
        # --------------------------------------------------------------------------------
        elif lower_message_text.startswith('/analyze'):
            parts = message_text.split()
            if len(parts) < 2:
                await send_telegram_reply(chat_id, "❌ Error: Please specify a symbol. Usage: `/analyze ETH/USDT`")
                return {"status": "ok", "message": "Missing symbol."}
                
            symbol = parts[1].upper().replace("/", "/").replace(" ", "") # Clean symbol input
            
            print(f"--- 🤖 TELEGRAM COMMAND RECEIVED from {chat_id}: /analyze {symbol}. ---")
            
            # 1. Send immediate response to let the user know the process started
            await send_telegram_reply(chat_id, f"🧠 Initiating **REAL ML Prediction** for **{symbol}**. This analysis is resource-intensive and may take a few seconds...")

            # 2. Execute the full prediction pipeline for the single symbol (using asyncio.to_thread)
            # This executes the synchronous function run_single_symbol_prediction in a separate 
            # thread, preventing the Uvicorn main loop from blocking.
            prediction_result = await asyncio.to_thread(run_single_symbol_prediction, symbol) 

            # 3. Process result and reply
            if prediction_result:
                metrics = prediction_result['metrics']
                
                # Check if a strong alert was found based on the 70% threshold
                if prediction_result['alert_found']:
                    response_text = (
                        f"✅ **PREDICCIÓN DE ALTA VENTAJA ENCONTRADA**\n"
                        f"------------------------------------\n"
                        f"**Activo:** {symbol}\n"
                        f"**Dirección:** {metrics['direction']}\n"
                        f"**Mejor Objetivo (R:R):** {metrics['best_target']} (Prob: {metrics['best_prob']:.2f}%)\n"
                        f"**Ventaja Neta:** +{metrics['max_advantage']:.2f}%\n"
                        f"**Hora:** {prediction_result['timestamp_local']} VET"
                    )
                else:
                    response_text = (
                        f"🟡 **Predicción de {symbol} Completada**\n"
                        f"------------------------------------\n"
                        f"**Dirección Sugerida:** {metrics['direction']}\n"
                        f"**Mejor Objetivo (R:R):** {metrics['best_target']} (Prob: {metrics['best_prob']:.2f}%)\n"
                        f"**Ventaja Neta:** +{metrics['max_advantage']:.2f}%\n"
                        f"**Conclusión:** La ventaja es inferior al umbral de 70%. No se recomienda la entrada."
                    )
            else:
                response_text = f"❌ Error: No se pudo obtener la predicción para {symbol}. Verifique que el símbolo sea válido o que la API esté accesible."
                
            await send_telegram_reply(chat_id, response_text)
            return {"status": "ok", "message": "Analyze command processed."}
            
        # --------------------------------------------------------------------------------
        # 3. /FEATURES COMMAND (Triggers full market feature report - Light Scan)
        # --------------------------------------------------------------------------------
        elif lower_message_text == '/features':
            print(f"--- 🤖 TELEGRAM COMMAND RECEIVED from {chat_id}: /features. Generating report... ---")
            
            await send_telegram_reply(chat_id, "⏳ Compiling latest market feature report...")
            
            # MOCK PLACEHOLDER FUNCTION (RETAINED FOR /FEATURES TEMPORARILY)
            features_summary = get_latest_features_summary(symbol="MARKET")
            
            await send_telegram_reply(chat_id, features_summary)
            return {"status": "ok", "message": "Features command processed."}

        # --------------------------------------------------------------------------------
        # 4. UNKNOWN COMMANDS
        # --------------------------------------------------------------------------------
        else:
            if lower_message_text.startswith('/'):
                await send_telegram_reply(chat_id, f"Unrecognized command: `{lower_message_text}`. Available commands:\n- `/scan`\n- `/analyze [symbol]`\n- `/features`")
            return {"status": "ok", "message": "Message received but no action taken."}
        
    except Exception as e:
        print(f"--- ❌ WEBHOOK PROCESSING ERROR for Chat ID {chat_id}: {e} ---")
        if chat_id:
            await send_telegram_reply(chat_id, "❌ Critical error during command processing. Check server logs.")
            
        return {"status": "error", "message": "Internal server error during webhook processing."}, 500

# --- MOCK PLACEHOLDER FUNCTION (RETAINED FOR /FEATURES TEMPORARILY) ---
def get_latest_features_summary(symbol: str = "MARKET") -> str:
    """Returns a mock summary of the latest features analysis for a given symbol."""
    if symbol.upper() == "MARKET":
        return (
            "📊 *Latest Market Feature Analysis (Mock Data)*\n\n"
            "**Volatility (ATR):** The 14-period Average True Range (ATR) is currently **0.95**, "
            "indicating low to moderate volatility across major pairs.\n"
            "**Momentum (RSI):** The Relative Strength Index (RSI) for most assets is around **48-55**, "
            "suggesting a neutral market bias. No overbought/oversold conditions.\n"
            "**Trend (MACD):** The MACD line has recently crossed below the signal line on 4-hour charts, "
            "hinting at a potential *short-term bearish reversal*.\n\n"
            "_Note: This data is simulated. Real data extraction from the scanner model must be implemented._"
        )
    else:
        return (
            f"🔍 *Analysis for {symbol.upper()} (Mock Data)*\n\n"
            f"**Current Price:** 1.0725\n"
            f"**RSI (14):** 68.2 (Approaching Overbought)\n"
            f"**Net Advantage:** N/A (Manual analysis does not calculate Net Advantage).\n"
            f"**Recommendation:** Pending full execution. Use `/scan` for automated signal search."
        )
# -----------------------------------------------------------------------------