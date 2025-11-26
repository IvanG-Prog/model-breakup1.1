import time
import threading
import sys
import os
import datetime 
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv 

load_dotenv() 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.model_pipeline.scanner_live import main as run_scanner_main 


# Scanner will run every 60 minutes (3600 seconds), matching the 1h timeframe
SCAN_INTERVAL_SECONDS = 3600
stop_event = threading.Event()

def scanner_scheduler():
    """Loop that runs the scanner continuously every 60 minutes (3600 seconds) 24/7 and forces an execution upon startup for immediate verification."""
    
    # FORCED EXECUTION ON STARTUP (always runs once)
    try:
        current_time_vet = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S VET')
        print(f"\n--- ⚙️ FORCED Execution on Startup - Local Time: {current_time_vet} ---")
        run_scanner_main()
        print("--- ✅ Initial Scan Completed. ---")
    except Exception as e:
        print(f"--- ❌ FATAL ERROR in initial scanner: {e} ---")

    
    while not stop_event.is_set():
        
        # Calculate next run time
        next_run_time_utc = datetime.datetime.utcnow() + datetime.timedelta(seconds=SCAN_INTERVAL_SECONDS)
        
        print(f"--- ⏳ Next 24/7 scan scheduled for: {next_run_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} ---")
        
        # Wait for the interval (or until stop_event is set)
        stop_event.wait(SCAN_INTERVAL_SECONDS)

        if not stop_event.is_set():
            try:
                current_time_vet = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S VET')
                print(f"\n--- ⚙️ Running scheduled scan - Local Time: {current_time_vet} ---")
                run_scanner_main()
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

@app.get("/run_scanner")
def run_scanner_api():
    """Endpoint to manually trigger the scanner and send an alert ONLY if a strong signal (>= 70%) is found."""
    try:
        run_scanner_main()
        return {"status": "success", "message": "Scanner executed on demand. Check Telegram for a signal (only if Net Advantage >= 70.0%)."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to run scanner: {str(e)}"}