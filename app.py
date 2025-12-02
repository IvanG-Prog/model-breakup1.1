import sys
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv 

# Load environment variables from .env file
load_dotenv() 

# Set up import paths for modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import scheduler functions (using the corrected names: start_scanner and stop_scanner)
from src.scheduler.background_scheduler import (
    start_scanner, 
    stop_scanner, 
    SCAN_INTERVAL_SECONDS
)

# Import the API router (from src/api/routes.py)
from src.api.routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifecycle function to start the scheduler thread on startup and stop it on shutdown."""
    
    # 1. Start the Scanner/Scheduler
    start_scanner(app)
    
    yield
    
    # 2. Stop the Scanner/Scheduler when the server shuts down
    stop_scanner()


app = FastAPI(
    title="ML Scanner & Scheduler",
    description="Web server that hosts and schedules the ML scanner continuously in the background.",
    version="1.0.0",
    lifespan=lifespan
)

# Include the external API routes for utility and webhook
app.include_router(api_router)

@app.get("/")
def read_root():
    """Status endpoint to check the service status."""
    return {
        "service_status": "Running",
        "scanner_status": f"Active, scanning automatically every {int(SCAN_INTERVAL_SECONDS / 60)} minutes (24/7).",
        "deployment": "Hugging Face Docker Space"
    }