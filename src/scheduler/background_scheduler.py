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
    # IMPORTANT: Importing synchronous functions (send_ping_message, send_proxy_keep_alive) 
    # and the main asynchronous runner (run_scanner_main).
    from src.model_pipeline.scanner_live import run_scanner_main, send_ping_message, send_proxy_keep_alive
except ImportError:
    print("❌ ERROR: Could not import necessary functions from scanner_live. Is the path correct?")
    # Placeholder definition in case of import failure
    async def run_scanner_main(source: str):
        print(f"--- FAILED RUNNER: Scheduled scan triggered ({source}) ---")
        return False
    # Placeholder definitions for synchronous functions
    def send_ping_message(message: str): 
        print(f"--- FAILED PING: Could not send ping: {message} ---")
        return False
    def send_proxy_keep_alive(): 
        print("--- FAILED KEEP ALIVE: Placeholder execution. ---")
        return False


# --- Async Runner Wrapper (Used by Scheduler and Manual API Trigger) ---
async def run_async_scanner(source: str) -> bool | None:
    """
    Executes the main scanner logic asynchronously.
    """
    try:
        # run_scanner_main is async due to internal logic, so it MUST be awaited.
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
        print("--- ✅ Report sent: Strong signal found. ---")
    else:
        # If no alert is found, send a guaranteed status message
        no_signal_message = f"📊 **DAILY MARKET REPORT ({report_time} VET)**\n\nNo high conviction signal (Net Advantage < 70%) was found in ETH/USDT at this time."
        
        try:
             # send_ping_message is synchronous and called directly.
             send_ping_message(no_signal_message)
        except Exception as e:
             # Critical log for network failure
             print(f"--- ❌ ERROR CRÍTICO RED: Falló el envío del mensaje de reporte diario a Render: {e} ---")

        print("--- 🟡 Report executed: No strong signal found. ---")


# --- Main Background Loop ---

async def background_scanner_loop():
    """
    The main infinite loop for the background scanner task.
    """
    print(f"--- 🟢 Scanner Loop Started. Interval: {int(SCAN_INTERVAL_SECONDS / 60)} minutes ---")
    
    # 1. Send initial ping message on startup for verification
    try:
        await asyncio.sleep(5) 
        # send_ping_message is synchronous and called directly.
        send_ping_message("🤖 Scanner Service Initialized! Running 24/7.")
        print("--- ✅ Initial startup message sent successfully. ---")
    except Exception as e:
        # Error correction: Removed incorrect 'await' that caused the 'object bool can't be used in await' warning
        print(f"--- ⚠️ WARNING: Initial startup message FAILED due to an error: {e}. Allowing scanner to continue. ---")
        
    # Main scan loop
    while True:
        await asyncio.sleep(1) 
        
        now_utc = datetime.now(timezone.utc)
        # Convert UTC to Venezuela Time (VET = UTC-4)
        now_vet = now_utc + timedelta(hours=-VENEZUELA_TZ_OFFSET)
        
        current_time_str = now_vet.strftime("%H:%M")
        current_minute = now_vet.minute
        current_second = now_vet.second

        # 0. HEARTBEAT LOG (Every minute at the 5th second)
        if current_second == 5:
            print(f"--- ❤️ HEARTBEAT: Loop Active at {now_vet.strftime('%H:%M:%S')} VET ---")
            
        # 1. Dedicated Keep-Alive Ping (Every 10 minutes: 0, 10, 20, 30, 40, 50)
        # Executes between second 0 and 4
        if current_minute % 10 == 0 and current_second < 5: 
            print(f"--- ⏱️ KEEP ALIVE: Pinging proxy at {current_time_str} VET... ---")
            try:
                # send_proxy_keep_alive is synchronous and called directly.
                send_proxy_keep_alive() 
            except Exception as e:
                # Catching any exception from the keep-alive function
                print(f"--- ❌ KEEP-ALIVE FAILED (Network Failure): {e} ---") 

        # 1.5. PRE-WARM PING (5 minutes before the hourly scan: 55)
        # Executes between second 0 and 4
        if current_minute == 55 and current_second < 5: 
            print(f"--- 🌡️ PRE-WARM PING: Waking proxy for next hour's scan at {current_time_str} VET... ---")
            try:
                # send_proxy_keep_alive is synchronous and called directly.
                send_proxy_keep_alive() 
            except Exception as e:
                # Catching any exception from the keep-alive function
                print(f"--- ❌ PRE-WARM FAILED: Could not wake proxy. ---") 
            
        
        # 2. Check for Hourly Scan Trigger and Daily Reports (UNIFIED CHECK)
        # Executes between second 0 and 4
        if current_minute == 0 and current_second < 5: 
            print(f"\n--- ⏳ HOURLY CHECK TRIGGER at {current_time_str} VET ---")
            
            if current_time_str in REPORT_TIMES_VET:
                # It's a critical report hour.
                await send_daily_report(current_time_str)
            else:
                # It's a normal scanning hour.
                alert_sent = await run_async_scanner(source='SCHEDULED_HOURLY')
                
                if alert_sent is True:
                    print("--- ✅ Scheduled Scan Complete: Alert Sent. ---")
                else:
                    # IF NO ALERT AND NOT A REPORT TIME: SEND HOURLY STATUS PING FOR CONFIRMATION
                    hourly_ping_message = f"🟢 **SCANNER STATUS ({current_time_str} VET)**\n\nNo high conviction signal (Net Advantage < 70%) was found in ETH/USDT."
                    try:
                        # send_ping_message is synchronous and called directly.
                        send_ping_message(hourly_ping_message)
                        print("--- 🟡 Hourly Status Ping Sent. ---")
                    except Exception as e:
                        # Critical log for network failure
                        print(f"--- ❌ CRITICAL NETWORK ERROR: Failed to send time status ping to Render: {e} ---")

        # Add a short sleep to avoid excessive CPU usage in the loop
        await asyncio.sleep(1) 

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