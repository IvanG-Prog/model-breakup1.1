import asyncio 
import os
import httpx 
from fastapi import APIRouter, Request, HTTPException

from src.scheduler.background_scheduler import run_async_scanner 
from src.model_pipeline.scanner_live import run_single_symbol_prediction 

# --- FastAPI Router Definition (CRITICAL FOR IMPORT) ---
router = APIRouter()

# --- Telegram Reply Utility (Fixed to use Proxy) ---

async def send_proxy_reply(chat_id: str, message: str):
    """
    Sends a message response back to a specific Telegram user/chat ID using the PROXY URL.
    Uses 'message' key in payload to comply with proxy requirements (fixed 422 error).
    """
    TELEGRAM_PROXY_URL = os.environ.get("TELEGRAM_PROXY_URL")
    if not TELEGRAM_PROXY_URL:
        print("--- ⚠️ TELEGRAM_PROXY_URL is not set. Cannot send user reply. ---")
        return False
        
    url = TELEGRAM_PROXY_URL 
    TIMEOUT_SECONDS = 30
    payload = {
        'chat_id': chat_id, 
        'message': message, 
        'parse_mode': 'Markdown'
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=TIMEOUT_SECONDS)
            response.raise_for_status() 
            print("✅ Telegram Reply sent successfully via Proxy.")
            return True
    except Exception as e:
        print(f"--- ❌ Error sending Telegram reply via Proxy: {e} ---")
        if isinstance(e, httpx.HTTPStatusError):
            print(f"--- ⚠️ Proxy server responded with body: {e.response.text} ---")
        return False


# --- Utility Endpoint ---
@router.get("/utility")
async def utility_endpoint(action: str):
    """
    Utility endpoint to manually trigger run_scan or test_connection tests.
    """
    if action == 'test_connection':
        return {"status": "success", "message": "✅ Core service is running and endpoints are accessible."}
            
    elif action == 'run_scan':
        # run_async_scanner es async y requiere 'await'
        print("--- 🤖 MANUAL API COMMAND: run_scan. Triggering immediate scan... ---")
        alert_sent = await run_async_scanner(source='MANUAL_API') 
        
        if alert_sent is True:
            return {"status": "success", "message": "✅ Strong signal found! Alert sent to Telegram."}
        elif alert_sent is False:
            return {"status": "info", "message": "🟡 Prediction found, but Net Advantage was below the 70.0% alert threshold. No message sent."}
        else:
            # alert_sent is None (error during execution)
            raise HTTPException(status_code=500, detail="Failed to run scanner due to an internal execution error.")
            
    else:
        raise HTTPException(status_code=400, detail="Invalid action specified. Use 'test_connection' or 'run_scan'.")


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
        # 1. /SCAN COMMAND 
        # --------------------------------------------------------------------------------
        if lower_message_text == '/scan':
            print(f"--- 🤖 TELEGRAM COMMAND RECEIVED from {chat_id}: /scan. Triggering immediate scan... ---")
            
            # Bot response (English) - First message
            await send_proxy_reply(chat_id, "🔍 Initiating full market scan. Please wait (this can take 30-60 seconds)...")

            # run_async_scanner is async and requires 'await'
            alert_found = await run_async_scanner(source='TELEGRAM_WEBHOOK') 
            
            # Bot response (English) - Second message
            if alert_found is True:
                response_text = "✅ Scan completed! A strong signal has been detected and sent to the alert channel."
            elif alert_found is False:
                response_text = "🟡 Scan completed. No strong signal (Net Advantage < 70%) found at this time."
            else:
                response_text = "❌ Scan failed due to an internal execution error. Check server logs."
                
            await send_proxy_reply(chat_id, response_text)
            return {"status": "ok", "message": "Scan command processed."}

        # --------------------------------------------------------------------------------
        # 2. /ANALYZE [SYMBOL] COMMAND 
        # --------------------------------------------------------------------------------
        elif lower_message_text.startswith('/analyze'):
            parts = message_text.split()
            if len(parts) < 2:
                # Bot response (English)
                await send_proxy_reply(chat_id, "❌ Error: Please specify a symbol. Usage: `/analyze ETH/USDT`")
                return {"status": "ok", "message": "Missing symbol."}
                
            symbol = parts[1].upper().replace("/", "/").replace(" ", "") 
            
            print(f"--- 🤖 TELEGRAM COMMAND RECEIVED from {chat_id}: /analyze {symbol}. ---")
            
            # Bot response (English) - Initial message
            await send_proxy_reply(chat_id, f"🧠 Initiating **REAL ML Prediction** for **{symbol}**. This analysis is resource-intensive and may take a few seconds...")

            prediction_result = await asyncio.to_thread(run_single_symbol_prediction, symbol) 

            # Process result and reply (Bot response in English)
            if prediction_result:
                metrics = prediction_result['metrics']
                
                # CONFIRMACIÓN DEL MENSAJE EN INGLÉS:
                if prediction_result['alert_found']:
                    response_text = (
                        f"✅ **HIGH ADVANTAGE PREDICTION FOUND**\n"
                        f"------------------------------------\n"
                        f"**Asset:** {symbol}\n"
                        f"**Direction:** {metrics['direction']}\n"
                        f"**Best Target (R:R):** {metrics['best_target']} (Prob: {metrics['best_prob']:.2f}%)\n"
                        f"**Net Advantage:** +{metrics['max_advantage']:.2f}%\n"
                        f"**Time:** {prediction_result['timestamp_local']} VET"
                    )
                else:
                    response_text = (
                        f"🟡 **{symbol} Prediction Complete**\n"
                        f"------------------------------------\n"
                        f"**Suggested Direction:** {metrics['direction']}\n"
                        f"**Best Target (R:R):** {metrics['best_target']} (Prob: {metrics['best_prob']:.2f}%)\n"
                        f"**Net Advantage:** +{metrics['max_advantage']:.2f}%\n"
                        f"**Conclusion:** Advantage is below the 70% threshold. Entry is not recommended."
                    )
            else:
                response_text = f"❌ Error: Could not retrieve prediction for {symbol}. Verify the symbol is valid or the API is accessible."
                
            await send_proxy_reply(chat_id, response_text)
            return {"status": "ok", "message": "Analyze command processed."}
            
        # --------------------------------------------------------------------------------
        # 3. /FEATURES COMMAND 
        # --------------------------------------------------------------------------------
        elif lower_message_text == '/features':
            print(f"--- 🤖 TELEGRAM COMMAND RECEIVED from {chat_id}: /features. Generating report... ---")
            
            # Bot response (English)
            await send_proxy_reply(chat_id, "⏳ Compiling latest market feature report...")
            
            # MOCK PLACEHOLDER FUNCTION (RETAINED FOR /FEATURES TEMPORARILY)
            features_summary = get_latest_features_summary(symbol="MARKET")
            
            await send_proxy_reply(chat_id, features_summary)
            return {"status": "ok", "message": "Features command processed."}

        # --------------------------------------------------------------------------------
        # 4. UNKNOWN COMMANDS
        # --------------------------------------------------------------------------------
        else:
            if lower_message_text.startswith('/'):
                # Bot response (English)
                await send_proxy_reply(chat_id, f"Unrecognized command: `{lower_message_text}`. Available commands:\n- `/scan`\n- `/analyze [symbol]`\n- `/features`")
            return {"status": "ok", "message": "Message received but no action taken."}
        
    except Exception as e:
        print(f"--- ❌ WEBHOOK PROCESSING ERROR for Chat ID {chat_id}: {e} ---")
        if chat_id:
            # Bot response (English)
            await send_proxy_reply(chat_id, "❌ Critical error during command processing. Check server logs.")
            
        return {"status": "error", "message": "Internal server error during webhook processing."}, 200

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