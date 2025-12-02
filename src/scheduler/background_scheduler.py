import asyncio
from typing import Optional
from fastapi import FastAPI
import sys
import os
from datetime import datetime, timezone, timedelta

# --- Constants ---
SCAN_INTERVAL_SECONDS = 3600
# Times for daily reports (in UTC-4/Venezuela time)
REPORT_TIMES_VET = ["08:00", "12:00", "17:00", "21:00"] 
VENEZUELA_TZ_OFFSET = 4 # UTC-4

# --- Global State ---
_scanner_task: Optional[asyncio.Task] = None

# --- Import Runner ---
# The path must be absolute to ensure it works in all contexts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    # IMPORTANT: Now importing the new function send_proxy_keep_alive
    from src.model_pipeline.scanner_live import run_scanner_main, send_ping_message, send_proxy_keep_alive
except ImportError:
    print("❌ ERROR: Could not import necessary functions from scanner_live. Is the path correct?")
    # Placeholder definition in case of import failure
    async def run_scanner_main(source: str):
        print(f"--- FAILED RUNNER: Scheduled scan triggered ({source}) ---")
        return False
    async def send_ping_message(message: str):
        print(f"--- FAILED PING: Could not send ping: {message} ---")
        return False
    # Placeholder for the new function
    async def send_proxy_keep_alive():
        print("--- FAILED KEEP ALIVE: Placeholder execution. ---")
        return False


# --- Async Runner Wrapper (Used by Scheduler and Manual API Trigger) ---

async def run_async_scanner(source: str) -> bool | None:
    """
    Executes the main scanner logic asynchronously.
    """
    try:
        # run_scanner_main is now async, so we call it with await
        alert_sent = await run_scanner_main(source) 
        return alert_sent
    except Exception as e:
        print(f"--- ❌ ASYNC EXECUTION ERROR in run_async_scanner: {e} ---")
        return None

# --- Daily Report Logic ---
async def send_daily_report(report_time: str):
    """Triggers a scan specifically for daily reports."""
    print(f"--- 📊 DAILY REPORT TRIGGER: Sending report for {report_time} VET... ---")
    
    # Execute the scan logic
    alert_sent = await run_async_scanner(source=f'REPORT_{report_time.replace(":", "")}')
    
    if alert_sent is True:
        # If alert was sent, the signal was strong enough
        print("--- ✅ Report sent: Strong signal found. ---")
    else:
        # If no alert, send a generic "no strong signal" report status
        no_signal_message = f"📊 **DAILY MARKET REPORT ({report_time} VET)**\n\nNo high conviction signal (Net Advantage < 70%) was found in ETH/USDT at this time.."
        await send_ping_message(no_signal_message)
        print("--- 🟡 Report sent: No strong signal found. ---")


# --- Main Background Loop ---

async def background_scanner_loop():
    """
    The main infinite loop for the background scanner task, handling both
    hourly checks, specific daily report times, and the 10-minute keep-alive ping.
    """
    print(f"--- 🟢 Scanner Loop Started. Interval: {int(SCAN_INTERVAL_SECONDS / 60)} minutes ---")
    
    # 1. Send initial ping message on startup for verification (NOW ASYNC)
    await send_ping_message("🤖 Scanner Service Initialized! Running 24/7.")
    
    # Main scan loop
    while True:
        await asyncio.sleep(1) # Check every second to be precise for report times
        
        now_utc = datetime.now(timezone.utc)
        # Convert UTC to Venezuela Time (VET = UTC-4)
        now_vet = now_utc + timedelta(hours=-VENEZUELA_TZ_OFFSET)
        
        current_time_str = now_vet.strftime("%H:%M")
        current_minute = now_vet.minute

        # 1. Dedicated Keep-Alive Ping (Every 10 minutes: 0, 10, 20, 30, 40, 50)
        if current_minute % 10 == 0 and now_vet.second < 5: 
             print("--- ⏱️ KEEP ALIVE: Pinging proxy service to prevent sleep... ---")
             await send_proxy_keep_alive() # Send silent ping to proxy
            
        # 2. Check for Hourly Scan Trigger (Only run the heavy scan once per interval)
        if current_minute == 0 and now_vet.second < 2: 
            print("\n--- ⏳ SCHEDULER TRIGGER: Starting scheduled scan... ---")
            await run_async_scanner(source='SCHEDULED')
            print("--- ✅ Scheduled Scan Complete. ---")
            
        # 3. Check for Daily Report Times
        if current_time_str in REPORT_TIMES_VET and current_minute == 0 and now_vet.second < 5:
            await send_daily_report(current_time_str)
            
        # Add a short sleep to avoid excessive CPU usage in the loop
        await asyncio.sleep(58) # Sleep for almost a minute before the next check

# --- Scheduler Control Functions ---

def start_scanner(app: FastAPI):
    """
    Starts the background scanner task.
    """
    global _scanner_task
    if _scanner_task and not _scanner_task.done():
        print("⚠️ Scanner is already running.")
        return

    _scanner_task = asyncio.create_task(background_scanner_loop())
    print("✅ Background Scanner Task created successfully.")

def stop_scanner():
    """
    Stops the background scanner task cleanly.
    """
    global _scanner_task
    if _scanner_task:
        _scanner_task.cancel()
        print("🛑 Background Scanner Task stopped.")
        _scanner_task = None