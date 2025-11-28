import time
import threading
import sys
import os
import datetime 
import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv 

load_dotenv() 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Necessary imports from scanner_live.py
from src.model_pipeline.scanner_live import main as run_scanner_main 
from src.model_pipeline.scanner_live import send_ping_message 
from src.model_pipeline.scanner_live import VENEZUELA_TZ_OFFSET 


# Scanner will run every 60 minutes (3600 seconds), matching the 1h timeframe
SCAN_INTERVAL_SECONDS = 3600
stop_event = threading.Event()

# --- ASYNC WRAPPER ---
def run_async_scanner():
    """Wrapper function to execute the async scanner main function synchronously."""
    try:
        return asyncio.run(run_scanner_main()) 
    except Exception as e:
        print(f"--- ❌ ASYNC EXECUTION ERROR in run_async_scanner: {e} ---")
        return None
# ---------------------

def scanner_scheduler():
    """Loop that runs the scanner continuously every 10 minutes (600 seconds) 24/7."""
    
    # FORCED EXECUTION ON STARTUP (always runs once)
    try:
        # Calculate local VET time
        current_time_vet = datetime.datetime.now() + VENEZUELA_TZ_OFFSET
        print(f"\n--- ⚙️ FORCED Execution on Startup - Local Time: {current_time_vet.strftime('%Y-%m-%d %H:%M:%S VET')} ---")
        run_async_scanner()
        print("--- ✅ Initial Scan Completed. ---")
    except Exception as e:
        print(f"--- ❌ FATAL ERROR in initial scanner: {e} ---")

    # The continuous scanning loop
    while not stop_event.is_set():
        
        # Calculate next run time (UTC for internal logging)
        next_run_time_utc = datetime.datetime.utcnow() + datetime.timedelta(seconds=SCAN_INTERVAL_SECONDS)
        
        print(f"--- ⏳ Next 24/7 scan scheduled for: {next_run_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} ---")
        
        # Wait for the interval
        stop_event.wait(SCAN_INTERVAL_SECONDS)

        if not stop_event.is_set():
            try:
                # Calculate local VET time for ping and log
                current_time_vet = datetime.datetime.now() + VENEZUELA_TZ_OFFSET
                
                # --- WAKE-UP PING MESSAGE (to ensure the system is active) ---
                print(f"--- 💡 Sending wake-up ping to Telegram... ---")
                ping_message = f"🚀 **Scanner Waking Up**\nNext scheduled scan starting now at {current_time_vet.strftime('%Y-%m-%d %H:%M:%S')} VET."
                send_ping_message(ping_message)
                # ---------------------------
                
                print(f"\n--- ⚙️ Running scheduled scan - Local Time: {current_time_vet.strftime('%Y-%m-%d %H:%M:%S VET')} ---")
                
                run_async_scanner()
                
                print("--- ✅ Scan completed. ---")
            except Exception as e:
                print(f"--- ❌ FATAL ERROR in scanner: {e} ---")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifecycle function to start the scheduler thread on startup and stop it on shutdown."""
    global scanner_thread
    scanner_thread = threading.Thread(target=scanner_scheduler)
    scanner_thread.start()
    
    yield
    
    stop_event.set()
    scanner_thread.join()
    print("--- Scheduler stopped ---")


app = FastAPI(
    title="ML Scanner & Scheduler",
    description="Web server that hosts and schedules the ML scanner continuously in the background.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    """Status endpoint to check the service status."""
    return {
        "service_status": "Running",
        "scanner_status": f"Active, scanning automatically every {int(SCAN_INTERVAL_SECONDS / 60)} minutes (24/7).",
        "deployment": "Hugging Face Docker Space"
    }

@app.get("/test_proxy_connection")
def test_proxy_connection_api():
    """Endpoint to manually test Proxy/Direct Telegram connectivity."""
    try:
        # Use the exposed ping function for simple connectivity test
        test_message = "🚀 [TEST] Successful Connectivity Check (Triggered via API)."
        
        # send_ping_message is a synchronous function that handles the network request.
        success = send_ping_message(test_message)
        
        if success:
            return {"status": "success", "message": "✅ Test alert successfully sent via configured Telegram method (Proxy or Direct API)."}
        else:
            return {"status": "error", "message": "❌ Test alert sending failed. Check environment variables and server logs for details."}
        
    except Exception as e:
        return {"status": "error", "message": f"❌ An unexpected error occurred during the test: {str(e)}"}

@app.get("/run_scanner")
def run_scanner_api():
    """Endpoint to manually trigger the scanner and send an alert ONLY if a strong signal (>= 70%) is found."""
    try:
        alert_sent = run_async_scanner()

        if alert_sent is True:
            return {"status": "success", "message": "✅ Strong signal found! Alert sent to Telegram (via Proxy/Direct API)."}
        elif alert_sent is False:
            return {"status": "info", "message": "🟡 Prediction found, but Net Advantage was below the 70.0% alert threshold. No message sent."}
        else:
             return {"status": "error", "message": "Failed to run scanner due to a threading/async execution error."}
             
    except Exception as e:
        return {"status": "error", "message": f"Failed to run scanner: {str(e)}"}