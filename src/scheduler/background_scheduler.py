import threading
import datetime 
import asyncio
import time
import os
import sys

# This module contains the core scheduling logic for the ML scanner.
# It runs the scanner continuously in a separate thread, manages the 
# 15-minute intervals, and handles the periodic status checks via Telegram.

# Import necessary functions from the scanner
from src.model_pipeline.scanner_live import run_scanner_main 
from src.model_pipeline.scanner_live import send_ping_message 
from src.model_pipeline.scanner_live import VENEZUELA_TZ_OFFSET 


# --- SCHEDULER CONFIGURATION ---
SCAN_INTERVAL_SECONDS = 900 # Scan every 15 minutes
stop_event = threading.Event()
scanner_thread = None


# --- ASYNC WRAPPER ---
def run_async_scanner(source: str) -> bool | None:
    """Wrapper function to execute the main async scanner function synchronously."""
    try:
        # Pass the 'source' argument to the main scanner function
        return asyncio.run(run_scanner_main(source=source)) 
    except Exception as e:
        print(f"--- ❌ ASYNC EXECUTION ERROR in run_async_scanner: {e} ---")
        return None
# ---------------------


def scanner_scheduler():
    """Loop that runs the scanner continuously every 15 minutes (900 seconds) 24/7."""
    
    # FORCED EXECUTION ON STARTUP (always runs once at the beginning)
    try:
        current_time_vet = datetime.datetime.now() + VENEZUELA_TZ_OFFSET
        
        # 1. SYSTEM STARTUP MESSAGE (TELEGRAM PING)
        startup_message = (
            f"🟢 **SYSTEM STARTUP**\n"
            f"Scheduler initialized and starting its first scan at {current_time_vet.strftime('%Y-%m-%d %H:%M:%S')} VET."
        )
        try:
            print("--- 💬 Attempting to send Telegram Startup Ping... ---")
            send_ping_message(startup_message) 
            print("--- ✅ Telegram Ping Sent. ---")
        except Exception as e:
            print(f"--- ⚠️ TELEGRAM PING FAILED ON STARTUP: {e}. Check .env configuration. Continuing scan... ---")

        # 2. FORCED INITIAL EXECUTION
        print(f"\n--- ⚙️ FORCED Execution on Startup - Local Time: {current_time_vet.strftime('%Y-%m-%d %H:%M:%S VET')} ---")
        run_async_scanner(source='SCHEDULED')
        print("--- ✅ Initial Scan Completed. ---")
        
    except Exception as e:
        print(f"--- ❌ FATAL ERROR in initial scanner: {e} ---")

    # The continuous scanning loop
    while not stop_event.is_set():
        
        # Calculate next run time (using UTC for internal logging)
        next_run_time_utc = datetime.datetime.utcnow() + datetime.timedelta(seconds=SCAN_INTERVAL_SECONDS)
        print(f"--- ⏳ Next 24/7 scan scheduled for: {next_run_time_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} ---")
        
        # Wait for the interval
        stop_event.wait(SCAN_INTERVAL_SECONDS)

        if not stop_event.is_set():
            try:
                current_time_vet = datetime.datetime.now() + VENEZUELA_TZ_OFFSET
                current_hour = current_time_vet.hour
                current_minute = current_time_vet.minute
                
                # --- STATUS CHECK LOGIC (Sends once per hour at 00 minutes) ---
                # Check if the current hour is 8 AM, 12 PM, or 8 PM (20:00) VET AND minutes are 00.
                if current_hour in [8, 12, 20] and current_minute == 0:
                    print(f"--- 💡 Sending scheduled STATUS CHECK to Telegram (8, 12, 20 VET)... ---")
                    status_message = (
                        f"🟢 **STATUS CHECK** (Scheduled)\n"
                        f"System is alive and starting scan at {current_time_vet.strftime('%Y-%m-%d %H:%M:%S')} VET."
                    )
                    try:
                        send_ping_message(status_message)
                    except Exception as e:
                        print(f"--- ⚠️ TELEGRAM STATUS CHECK FAILED: {e}. Check .env configuration. ---")

                else:
                    print(f"--- 💡 Thread is alive. Next scan starting now (No Telegram status message sent). ---")
                # ---------------------------
                
                print(f"\n--- ⚙️ Running scheduled scan - Local Time: {current_time_vet.strftime('%Y-%m-%d %H:%M:%S VET')} ---")
                
                # Execute the scheduled scan
                run_async_scanner(source='SCHEDULED')
                
                print("--- ✅ Scan completed. ---")
            except Exception as e:
                print(f"--- ❌ FATAL ERROR in scanner: {e} ---")


def start_scheduler():
    """Initializes and starts the scheduler thread."""
    global scanner_thread
    scanner_thread = threading.Thread(target=scanner_scheduler)
    scanner_thread.start()
    
def stop_scheduler():
    """Stops the scheduler thread."""
    if scanner_thread and scanner_thread.is_alive():
        stop_event.set()
        scanner_thread.join()
        print("--- Scheduler stopped ---")