#!/usr/bin/env python
import os
import sys
import time
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Any
from crewai.flow import Flow, start, listen
from crewai import LLM
from collections import deque

# ===== Load environment variables =====
load_dotenv()

# ===== Fix Python path for imports =====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ===== Enhanced Rate Limiter (Prevents Rate Limit & Quota Errors) =====
class EnhancedRateLimiter:
    def __init__(self, max_requests_per_day: int = 200, max_tokens_per_minute: int = 5000):
        """
        Enhanced rate limiter that tracks both daily request counts and token usage.
        Gemini free tier: 250 requests/day per model - using 200 as safe limit.
        """
        self.token_usage = deque()  # List of (timestamp, tokens_used) tuples
        self.request_timestamps = deque()  # List of request timestamps for daily tracking
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_requests_per_day = max_requests_per_day
        self.estimated_tokens_per_call = 500
        self.last_request_time = None
        self.daily_reset_date = datetime.now().date()
    
    def _reset_if_new_day(self):
        """Reset daily counters if it's a new day."""
        today = datetime.now().date()
        if today != self.daily_reset_date:
            self.request_timestamps.clear()
            self.daily_reset_date = today
            print(f"📅 New day detected - Daily request counter reset")
    
    def wait_if_needed(self, estimated_tokens: int = None):
        """
        Wait if we're approaching token or daily request limits.
        """
        self._reset_if_new_day()
        
        if estimated_tokens is None:
            estimated_tokens = self.estimated_tokens_per_call
        
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        one_day_ago = now - timedelta(days=1)
        
        # Remove old token usage entries
        while self.token_usage and self.token_usage[0][0] < one_minute_ago:
            self.token_usage.popleft()
        
        # Remove old request timestamps
        while self.request_timestamps and self.request_timestamps[0] < one_day_ago:
            self.request_timestamps.popleft()
        
        # Check daily request limit (MOST IMPORTANT for free tier)
        daily_requests = len(self.request_timestamps)
        if daily_requests >= self.max_requests_per_day:
            # Calculate time until oldest request expires
            if self.request_timestamps:
                oldest_time = self.request_timestamps[0]
                wait_seconds = (one_day_ago + timedelta(days=1) - now).total_seconds() + 60  # 1 min buffer
                wait_hours = wait_seconds / 3600
                print(f"🚫 Daily quota limit reached ({daily_requests}/{self.max_requests_per_day} requests)")
                print(f"⏸️  Must wait {wait_hours:.1f} hours until quota resets...")
                print(f"💡 Consider reducing number of argument rounds or waiting until tomorrow.")
                raise Exception(f"Daily quota exhausted: {daily_requests}/{self.max_requests_per_day} requests used. Please wait or reduce trial complexity.")
            else:
                # Safety wait
                print(f"⏸️  Daily quota protection: waiting 60s...")
                time.sleep(60)
        
        # Check token limit per minute
        total_tokens = sum(tokens for _, tokens in self.token_usage)
        if total_tokens + estimated_tokens > self.max_tokens_per_minute:
            if self.token_usage:
                oldest_time = self.token_usage[0][0]
                wait_seconds = 60 - (now - oldest_time).seconds + 5
                print(f"⏸️  Token rate limit protection: waiting {wait_seconds}s... (Used: {total_tokens}/{self.max_tokens_per_minute} tokens)")
                time.sleep(wait_seconds)
                self.token_usage.clear()
            else:
                print(f"⏸️  Token rate limit protection: waiting 10s...")
                time.sleep(10)
        
        # Record this call
        self.token_usage.append((now, estimated_tokens))
        self.request_timestamps.append(now)
        self.last_request_time = now
        
        # Add delay between calls to prevent burst (reduced to save quota)
        if self.last_request_time:
            time_since_last = (now - self.last_request_time).total_seconds()
            if time_since_last < 2.0:  # Minimum 2 seconds between requests
                wait = 2.0 - time_since_last
                time.sleep(wait)
        else:
            time.sleep(1.0)  # Initial delay
    
    def get_daily_usage(self) -> tuple:
        """Return (used, limit) for daily requests."""
        self._reset_if_new_day()
        return (len(self.request_timestamps), self.max_requests_per_day)

# Create global rate limiter instance
token_rate_limiter = EnhancedRateLimiter(max_requests_per_day=200, max_tokens_per_minute=5000)

# ===== Callback Handler (Requirement #4: Monitoring & Logging) =====
class CourtRoomCallbackHandler:
    """Callback handler for monitoring courtroom agent activities."""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"courtroom_session_{self.session_id}.log"
        self.execution_log = []
        
    def log_event(self, event_type: str, details: Dict[str, Any]):
        """Log any event with details."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            **details
        }
        self.execution_log.append(entry)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
        
        # Console output with emojis
        emoji_map = {
            "CREW_START": "📋",
            "CREW_COMPLETE": "✅",
            "ARGUMENT": "⚖️",
            "A2A_COMMUNICATION": "🤝",
            "TOOL_USE": "🔧",
            "RATE_LIMIT": "⏱️",
            "ERROR": "❌"
        }
        emoji = emoji_map.get(event_type, "📌")
        
        if event_type == "CREW_START":
            print(f"\n{emoji} [CREW START] {details.get('crew_name', 'Unknown')}")
        elif event_type == "CREW_COMPLETE":
            print(f"{emoji} [CREW COMPLETE] {details.get('crew_name', 'Unknown')} - Duration: {details.get('duration', 0):.2f}s")
        elif event_type == "ARGUMENT":
            print(f"{emoji} [ARGUMENT {details.get('round', 0)}] {details.get('side', 'Unknown')}")
        elif event_type == "A2A_COMMUNICATION":
            print(f"{emoji} [A2A] {details.get('from_agent', 'Agent')} → Legal Advisor")
        elif event_type == "TOOL_USE":
            print(f"{emoji} [TOOL] {details.get('agent', 'Agent')} used {details.get('tool', 'tool')}")
        elif event_type == "RATE_LIMIT":
            print(f"{emoji} [RATE LIMIT] {details.get('crew_name', 'Unknown')} - Attempt {details.get('attempt', 1)}")
        elif event_type == "ERROR":
            print(f"{emoji} [ERROR] {details.get('message', 'Unknown error')}")
    
    def generate_summary(self) -> str:
        """Generate execution summary."""
        summary = f"""
{'='*60}
COURTROOM SESSION SUMMARY
{'='*60}
Session ID: {self.session_id}
Total Events: {len(self.execution_log)}

Event Breakdown:
- Crew Executions: {len([e for e in self.execution_log if e['event'] == 'CREW_START'])}
- Completed: {len([e for e in self.execution_log if e['event'] == 'CREW_COMPLETE'])}
- Arguments: {len([e for e in self.execution_log if e['event'] == 'ARGUMENT'])}
- A2A Communications: {len([e for e in self.execution_log if e['event'] == 'A2A_COMMUNICATION'])}
- Tool Uses: {len([e for e in self.execution_log if e['event'] == 'TOOL_USE'])}
- Rate Limit Hits: {len([e for e in self.execution_log if e['event'] == 'RATE_LIMIT'])}
- Errors: {len([e for e in self.execution_log if e['event'] == 'ERROR'])}

Log file: {self.log_file}
{'='*60}
"""
        return summary

# Initialize global callback handler
callback_handler = CourtRoomCallbackHandler()

# ===== Set up Gemini LLM =====
# Using Gemini 1.5 Flash for fast, cost-effective responses
llm = LLM(
    model="gemini/gemini-2.5-flash",  # ✅ Changed to gemini-1.5-pro (separate quota pool)
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    max_tokens=512,  # ✅ Increased since Gemini has better rate limits
)

# ===== Import A2A Module (Requirement #5) =====
try:
    from src.courtroom.tools.a2a_protocol import a2a_advisor, a2a_consultation_tool, log_a2a_communication
    print("✅ A2A Protocol loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: A2A module not found.")
    a2a_advisor = None
    a2a_consultation_tool = None

# ===== Import all Crew Modules =====
from src.courtroom.crews.case_management.case_management import CaseManagement
from src.courtroom.crews.prosecution.prosecution import Prosecution
from src.courtroom.crews.defense.defense import Defense
from src.courtroom.crews.witness.witness import Witness
from src.courtroom.crews.jury.jury import Jury
from src.courtroom.crews.judge.judge import Judge
from src.courtroom.crews.reporter.reporter import Reporter

# ===== Argument Structure (Pydantic for Structured Output - Requirement #3) =====
class Argument(BaseModel):
    """Structured argument model."""
    round_number: int
    side: str
    argument_type: str
    content: str
    legal_citations: List[str] = []
    timestamp: str
    used_a2a: bool = False

class TrialTranscript(BaseModel):
    """Complete trial transcript with structured data."""
    case_name: str
    session_id: str
    arguments: List[Argument] = []
    witness_testimonies: List[str] = []
    jury_verdict: str = ""
    judge_verdict: str = ""
    final_summary: str = ""
    a2a_consultations: int = 0

# ===== Define Courtroom State (Requirement #1: Context Sharing) =====
class CourtroomState(BaseModel):
    case_name: str = "Rajkumar vs. The State of Uttar Pradesh"
    case_summary: str = ""
    prosecution_opening: str = ""
    defense_opening: str = ""
    arguments: List[Argument] = []
    witness_testimony: str = ""
    prosecution_closing: str = ""
    defense_closing: str = ""
    jury_verdict: str = ""
    judge_final_verdict: str = ""
    final_report: str = ""
    transcript: TrialTranscript = None
    num_rounds: int = 5  # ✅ Added: Configurable argument rounds

# ===== Define Main Courtroom Flow =====
class CourtroomFlow(Flow[CourtroomState]):
    """Main AI Courtroom Simulation Flow with configurable Argument Rounds"""

    def __init__(self, case_name: str = None, num_rounds: int = 5):
        super().__init__()
        
        # Set case name if provided
        if case_name:
            self.state.case_name = case_name
            # Format case name properly
            if " vs. " not in case_name and " v. " not in case_name:
                self.state.case_name = f"{case_name} vs. The State"
        
        # Set number of rounds
        self.state.num_rounds = num_rounds
        
        self.state.transcript = TrialTranscript(
            case_name=self.state.case_name,
            session_id=callback_handler.session_id
        )

    def _run_crew_with_logging(self, crew_class, crew_name: str, inputs: Dict[str, Any], max_retries=3) -> Any:
        """Helper to run crew with logging, rate limiting, and retry on rate limit/quota errors."""
        
        for attempt in range(max_retries):
            try:
                # ✅ RATE LIMIT PROTECTION - Check daily quota before making request
                try:
                    token_rate_limiter.wait_if_needed()
                except Exception as quota_error:
                    if "Daily quota exhausted" in str(quota_error):
                        print(f"\n{'='*60}")
                        print(f"🚫 QUOTA EXHAUSTED - Cannot proceed")
                        print(f"{'='*60}")
                        used, limit = token_rate_limiter.get_daily_usage()
                        print(f"Daily requests used: {used}/{limit}")
                        print(f"\n💡 Solutions:")
                        print(f"   1. Wait until tomorrow (quota resets daily)")
                        print(f"   2. Reduce number of argument rounds (currently: {self.state.num_rounds})")
                        print(f"   3. Upgrade to paid Gemini API plan")
                        print(f"{'='*60}\n")
                        raise
                    else:
                        raise
                
                start_time = time.time()
                callback_handler.log_event("CREW_START", {
                    "crew_name": crew_name,
                    "inputs": str(inputs)[:200],
                    "attempt": attempt + 1
                })
                
                result = crew_class().crew().kickoff(inputs=inputs)
                duration = time.time() - start_time
                callback_handler.log_event("CREW_COMPLETE", {
                    "crew_name": crew_name,
                    "duration": duration,
                    "output_length": len(str(result.raw))
                })
                
                # ✅ SUCCESS - Reduced cooldown to save quota
                wait_time = 5 + (attempt * 2)  # Reduced from 30s to 5s base
                used, limit = token_rate_limiter.get_daily_usage()
                print(f"✅ {crew_name} completed (Daily quota: {used}/{limit})")
                if wait_time > 0:
                    print(f"⏸️  Cooldown: waiting {wait_time}s...")
                    time.sleep(wait_time)
                
                return result
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a quota/rate limit error (429 or RESOURCE_EXHAUSTED)
                is_quota_error = (
                    "429" in error_str or 
                    "RESOURCE_EXHAUSTED" in error_str or 
                    "quota" in error_str.lower() or
                    "rate_limit" in error_str.lower() or
                    "rate limit" in error_str.lower()
                )
                
                # Check if quota is EXHAUSTED (not just rate-limited)
                quota_exhausted = (
                    "exceeded your current quota" in error_str.lower() or
                    "quota exceeded" in error_str.lower() or
                    "free_tier_requests" in error_str.lower()
                )
                
                if is_quota_error:
                    callback_handler.log_event("RATE_LIMIT", {
                        "crew_name": crew_name,
                        "attempt": attempt + 1,
                        "error": error_str[:200],
                        "quota_exhausted": quota_exhausted
                    })
                    
                    # If quota is truly exhausted, stop immediately
                    if quota_exhausted:
                        print(f"\n{'='*60}")
                        print(f"🚫 QUOTA EXHAUSTED - Cannot proceed")
                        print(f"{'='*60}")
                        used, limit = token_rate_limiter.get_daily_usage()
                        print(f"Daily requests tracked: {used}/{limit}")
                        print(f"\n❌ The Gemini API quota for this model is exhausted.")
                        print(f"   Error: {error_str[:300]}")
                        print(f"\n💡 Solutions:")
                        print(f"   1. Wait until tomorrow (quota resets daily at midnight UTC)")
                        print(f"   2. Try a different Gemini model (we just switched to gemini-1.5-pro)")
                        print(f"   3. Check your quota at: https://ai.dev/usage?tab=rate-limit")
                        print(f"   4. Upgrade to a paid API plan for higher limits")
                        print(f"{'='*60}\n")
                        raise Exception(f"Quota exhausted for Gemini API. Please wait until quota resets or upgrade your plan.")
                    
                    # Otherwise, it's just rate limiting - retry with backoff
                    if attempt < max_retries - 1:
                        # Extract wait time from error message (Gemini provides retry delay)
                        match = re.search(r'retry in (\d+\.?\d*)s', error_str, re.IGNORECASE)
                        if match:
                            wait_time = float(match.group(1))
                        else:
                            # Default exponential backoff for rate limit errors
                            wait_time = 30 * (2 ** attempt)  # 30s, 60s, 120s
                        
                        wait_time += 10  # Add buffer
                        used, limit = token_rate_limiter.get_daily_usage()
                        
                        print(f"\n⚠️  Rate limit hit for {crew_name} (not quota exhausted)")
                        print(f"   Daily quota: {used}/{limit} requests")
                        print(f"   Waiting {wait_time:.0f}s before retry {attempt + 2}/{max_retries}...\n")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"\n❌ Max retries reached for {crew_name}")
                        used, limit = token_rate_limiter.get_daily_usage()
                        print(f"   Final daily quota: {used}/{limit} requests")
                        print(f"\n💡 The Gemini API free tier has a 250 requests/day limit.")
                        print(f"   Consider:")
                        print(f"   - Reducing argument rounds (currently: {self.state.num_rounds})")
                        print(f"   - Waiting until tomorrow for quota reset")
                        print(f"   - Using a paid API plan\n")
                        raise
                else:
                    # Non-quota error
                    callback_handler.log_event("ERROR", {
                        "crew_name": crew_name,
                        "error": str(e)[:200]
                    })
                    raise
        
        raise Exception(f"Failed to execute {crew_name} after {max_retries} attempts")

    def _maybe_use_a2a(self, agent_name: str, round_num: int) -> str:
        """Optionally use A2A consultation (Requirement #5)"""
        # Use A2A on strategic rounds (middle and later rounds)
        num_rounds = self.state.num_rounds
        # Calculate A2A rounds: if 5 rounds, use round 3; if 10 rounds, use 3, 6, 9
        if num_rounds <= 5:
            a2a_rounds = [3] if num_rounds >= 3 else []
        elif num_rounds <= 7:
            a2a_rounds = [3, 6] if num_rounds >= 6 else [3]
        else:
            a2a_rounds = [3, 6, 9] if num_rounds >= 9 else [3, 6]
        
        if a2a_advisor and round_num in a2a_rounds:
            query = f"As {agent_name}, what's the strongest legal strategy for round {round_num}?"
            log_a2a_communication(agent_name, query, callback_handler)
            advice = a2a_advisor.consult(query, {
                "round": round_num,
                "case": self.state.case_name
            })
            self.state.transcript.a2a_consultations += 1
            return f"\n\n[A2A ADVISOR CONSULTATION]\n{advice}\n"
        return ""

    @start()
    def manage_case(self):
        """Step 1: Load case data"""
        print("\n" + "="*80)
        print("⚖️  COURTROOM SIMULATION - TRIAL BEGINS")
        print("="*80 + "\n")
        print("📁 Stage 1: Case Management - Loading case files...\n")
        
        # Extract case name from state (handle "vs." format)
        case_name_for_tool = self.state.case_name.split(" vs. ")[0].split(" v. ")[0].strip()
        
        result = self._run_crew_with_logging(CaseManagement, "CaseManagement", {
            "case_name": case_name_for_tool  # ✅ Use case name from user input
        })
        self.state.case_summary = str(result.raw)
        print("✅ Case loaded\n")
        time.sleep(10)  # Wait before next step

    @listen(manage_case)
    def opening_statements(self):
        """Step 2: Opening Statements"""
        print("\n" + "="*80)
        print("🎤 Stage 2: OPENING STATEMENTS")
        print("="*80 + "\n")
        
        # Prosecution opening
        print("🧑‍⚖️  Prosecution presents opening statement...\n")
        result = self._run_crew_with_logging(Prosecution, "Prosecution_Opening", {
            "case_summary": self.state.case_summary,
            "task_type": "opening_statement"
        })
        self.state.prosecution_opening = result.raw
        
        opening_arg = Argument(
            round_number=0,
            side="prosecution",
            argument_type="opening",
            content=result.raw,
            timestamp=datetime.now().isoformat()
        )
        self.state.arguments.append(opening_arg)
        self.state.transcript.arguments.append(opening_arg)
        callback_handler.log_event("ARGUMENT", {"round": 0, "side": "prosecution", "type": "opening"})
        
        # Defense opening
        print("\n👨‍💼 Defense presents opening statement...\n")
        result = self._run_crew_with_logging(Defense, "Defense_Opening", {
            "case_summary": self.state.case_summary,
            "prosecution_opening": self.state.prosecution_opening,
            "prosecution_arguments": self.state.prosecution_opening,  # ✅ Added: Defense task expects this
            "task_type": "opening_statement"
        })
        self.state.defense_opening = result.raw
        
        opening_arg = Argument(
            round_number=0,
            side="defense",
            argument_type="opening",
            content=result.raw,
            timestamp=datetime.now().isoformat()
        )
        self.state.arguments.append(opening_arg)
        self.state.transcript.arguments.append(opening_arg)
        callback_handler.log_event("ARGUMENT", {"round": 0, "side": "defense", "type": "opening"})

    @listen(opening_statements)
    def argument_rounds(self):
        """Step 3: Argument Rounds (Configurable)"""
        num_rounds = self.state.num_rounds
        print("\n" + "="*80)
        print(f"⚔️  Stage 3: ARGUMENT ROUNDS ({num_rounds} Rounds)")
        print("="*80 + "\n")
        
        for round_num in range(1, num_rounds + 1):  # ✅ Use configurable number of rounds
            print(f"\n{'─'*80}")
            print(f"🔵 ROUND {round_num}/{num_rounds}")
            print(f"{'─'*80}\n")
            
            previous_args = "\n\n".join([
                f"[{arg.side.upper()} - Round {arg.round_number}]: {arg.content[:200]}..."
                for arg in self.state.arguments[-3:]  # ✅ Reduced context
            ])
            
            # PROSECUTION
            print(f"🧑‍⚖️  Prosecution - Argument {round_num}...\n")
            a2a_advice = self._maybe_use_a2a("Prosecution", round_num)
            
            result = self._run_crew_with_logging(Prosecution, f"Prosecution_Round_{round_num}", {
                "case_summary": self.state.case_summary[:500],  # ✅ Truncated context
                "round_number": round_num,
                "previous_arguments": previous_args,
                "a2a_advice": a2a_advice,
                "task_type": "argument"
            })
            
            prosecution_arg = Argument(
                round_number=round_num,
                side="prosecution",
                argument_type="argument",
                content=result.raw,
                timestamp=datetime.now().isoformat(),
                used_a2a=bool(a2a_advice)
            )
            self.state.arguments.append(prosecution_arg)
            self.state.transcript.arguments.append(prosecution_arg)
            callback_handler.log_event("ARGUMENT", {
                "round": round_num,
                "side": "prosecution",
                "type": "argument",
                "used_a2a": bool(a2a_advice)
            })
            
            # DEFENSE
            print(f"\n👨‍💼 Defense - Rebuttal to Round {round_num}...\n")
            a2a_advice = self._maybe_use_a2a("Defense", round_num)
            
            # Collect prosecution arguments for template
            prosecution_args_round = "\n".join([arg.content[:200] for arg in self.state.arguments if arg.side == "prosecution"])
            
            result = self._run_crew_with_logging(Defense, f"Defense_Round_{round_num}", {
                "case_summary": self.state.case_summary[:500],  # ✅ Truncated context
                "round_number": round_num,
                "prosecution_argument": prosecution_arg.content[:300],  # ✅ Truncated
                "prosecution_arguments": prosecution_args_round,  # ✅ Added: Defense task expects this
                "previous_arguments": previous_args,
                "a2a_advice": a2a_advice,
                "task_type": "rebuttal"
            })
            
            defense_arg = Argument(
                round_number=round_num,
                side="defense",
                argument_type="rebuttal",
                content=result.raw,
                timestamp=datetime.now().isoformat(),
                used_a2a=bool(a2a_advice)
            )
            self.state.arguments.append(defense_arg)
            self.state.transcript.arguments.append(defense_arg)
            callback_handler.log_event("ARGUMENT", {
                "round": round_num,
                "side": "defense",
                "type": "rebuttal",
                "used_a2a": bool(a2a_advice)
            })
            
            print(f"✅ Round {round_num} complete")

        print("\n" + "="*80)
        print(f"🏁 ALL {num_rounds} ROUNDS COMPLETED!")
        print(f"   Total A2A consultations: {self.state.transcript.a2a_consultations}")
        print("="*80 + "\n")

    @listen(argument_rounds)
    def witness_examination(self):
        """Step 4: Witness Testimony"""
        print("\n" + "="*80)
        print("🧍 Stage 4: WITNESS EXAMINATION")
        print("="*80 + "\n")
        
        # Collect all arguments for template variables
        prosecution_args = "\n".join([arg.content[:200] for arg in self.state.arguments if arg.side == "prosecution"])
        defense_args = "\n".join([arg.content[:200] for arg in self.state.arguments if arg.side == "defense"])
        
        result = self._run_crew_with_logging(Witness, "Witness", {
            "case_summary": self.state.case_summary[:500],
            "defense_arguments": defense_args,  # ✅ Added: Witness task expects this
            "arguments_summary": "\n".join([f"{arg.side}: {arg.content[:150]}" for arg in self.state.arguments[-3:]])
        })
        self.state.witness_testimony = result.raw
        self.state.transcript.witness_testimonies.append(result.raw)

    @listen(witness_examination)
    def closing_arguments(self):
        """Step 5: Closing Arguments"""
        print("\n" + "="*80)
        print("🎯 Stage 5: CLOSING ARGUMENTS")
        print("="*80 + "\n")
        
        # Prosecution closing
        print("🧑‍⚖️  Prosecution presents closing argument...\n")
        result = self._run_crew_with_logging(Prosecution, "Prosecution_Closing", {
            "case_summary": self.state.case_summary[:500],
            "all_arguments": f"Total: {len(self.state.arguments)} arguments",  # ✅ Simplified
            "witness_testimony": self.state.witness_testimony[:300],
            "task_type": "closing_argument"
        })
        self.state.prosecution_closing = result.raw
        
        closing_arg = Argument(
            round_number=99,
            side="prosecution",
            argument_type="closing",
            content=result.raw,
            timestamp=datetime.now().isoformat()
        )
        self.state.arguments.append(closing_arg)
        self.state.transcript.arguments.append(closing_arg)
        
        # Defense closing
        print("\n👨‍💼 Defense presents closing argument...\n")
        # Collect prosecution arguments for template
        prosecution_args_closing = "\n".join([arg.content[:200] for arg in self.state.arguments if arg.side == "prosecution"])
        
        result = self._run_crew_with_logging(Defense, "Defense_Closing", {
            "case_summary": self.state.case_summary[:500],
            "prosecution_arguments": prosecution_args_closing,  # ✅ Added: Defense task expects this
            "all_arguments": f"Total: {len(self.state.arguments)} arguments",
            "witness_testimony": self.state.witness_testimony[:300] if self.state.witness_testimony else "No witness testimony available",
            "prosecution_closing": self.state.prosecution_closing[:300],
            "task_type": "closing_argument"
        })
        self.state.defense_closing = result.raw
        
        closing_arg = Argument(
            round_number=99,
            side="defense",
            argument_type="closing",
            content=result.raw,
            timestamp=datetime.now().isoformat()
        )
        self.state.arguments.append(closing_arg)
        self.state.transcript.arguments.append(closing_arg)

    @listen(closing_arguments)
    def jury_deliberation(self):
        """Step 6: Jury Deliberation"""
        print("\n" + "="*80)
        print("👥 Stage 6: JURY DELIBERATION")
        print("="*80 + "\n")
        
        # Collect all arguments for template variables
        prosecution_args = "\n".join([arg.content[:200] for arg in self.state.arguments if arg.side == "prosecution"])
        defense_args = "\n".join([arg.content[:200] for arg in self.state.arguments if arg.side == "defense"])
        
        result = self._run_crew_with_logging(Jury, "Jury", {
            "case_summary": self.state.case_summary[:500],
            "prosecution_arguments": prosecution_args,  # ✅ Added: Jury task expects this
            "defense_arguments": defense_args,  # ✅ Added: Jury task expects this
            "witness_testimony": self.state.witness_testimony[:300] if self.state.witness_testimony else "No witness testimony available",
            "all_arguments": f"Review {len(self.state.arguments)} arguments from trial"
        })
        self.state.jury_verdict = result.raw
        self.state.transcript.jury_verdict = result.raw

    @listen(jury_deliberation)
    def judge_decision(self):
        """Step 7: Judge Final Verdict"""
        print("\n" + "="*80)
        print("⚖️  Stage 7: JUDGE'S FINAL VERDICT")
        print("="*80 + "\n")
        
        result = self._run_crew_with_logging(Judge, "Judge", {
            "case_summary": self.state.case_summary[:500],
            "jury_verdict": self.state.jury_verdict,
            "trial_summary": f"Total arguments: {len(self.state.arguments)}"
        })
        self.state.judge_final_verdict = result.raw
        self.state.transcript.judge_verdict = result.raw

    @listen(judge_decision)
    def report_generation(self):
        """Step 8: Final Report Generation"""
        print("\n" + "="*80)
        print("📰 Stage 8: GENERATING FINAL REPORT")
        print("="*80 + "\n")
        
        # Collect all arguments for template variables - FULL CONTENT for detailed summaries
        prosecution_args_full = "\n\n".join([
            f"=== ROUND {arg.round_number} ({arg.argument_type.upper()}) ===\n{arg.content}"
            for arg in self.state.arguments if arg.side == "prosecution"
        ])
        defense_args_full = "\n\n".join([
            f"=== ROUND {arg.round_number} ({arg.argument_type.upper()}) ===\n{arg.content}"
            for arg in self.state.arguments if arg.side == "defense"
        ])
        
        # Create detailed argument breakdown for reporter
        argument_breakdown = "\n\n".join([
            f"ROUND {arg.round_number} - {arg.side.upper()} ({arg.argument_type}):\n"
            f"Content: {arg.content[:500]}...\n"
            f"A2A Consultation Used: {'Yes' if arg.used_a2a else 'No'}\n"
            f"Timestamp: {arg.timestamp}\n"
            for arg in sorted(self.state.arguments, key=lambda x: (x.round_number, 0 if x.side == "prosecution" else 1))
        ])
        
        result = self._run_crew_with_logging(Reporter, "Reporter", {
            "case_summary": self.state.case_summary[:1000],  # ✅ Increased for better context
            "prosecution_arguments": prosecution_args_full,  # ✅ Full content for detailed summaries
            "defense_arguments": defense_args_full,  # ✅ Full content for detailed summaries
            "argument_breakdown": argument_breakdown,  # ✅ NEW: Detailed breakdown for reporter
            "witness_testimony": self.state.witness_testimony[:500] if self.state.witness_testimony else "No witness testimony available",
            "jury_verdict": self.state.jury_verdict[:500] if self.state.jury_verdict else "No jury verdict available",
            "judge_final_verdict": self.state.judge_final_verdict[:500] if self.state.judge_final_verdict else "No judge verdict available",
            "total_arguments": len(self.state.arguments),
            "num_rounds": self.state.num_rounds
        })
        self.state.final_report = result.raw
        self.state.transcript.final_summary = result.raw
        
        self._save_all_outputs()
        
        print("\n" + "="*80)
        print("🎉 TRIAL COMPLETE!")
        print("="*80)
        print(callback_handler.generate_summary())
        
        if self.state.transcript.a2a_consultations > 0:
            print(f"\n🤝 A2A Protocol Usage: {self.state.transcript.a2a_consultations} consultations")

    def _save_all_outputs(self):
        """Save all outputs in structured formats (Requirement #3)."""
        output_dir = Path(os.getenv("REPORT_OUTPUT_DIR", "./reports"))
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Markdown report
        report_path = output_dir / f"trial_report_{timestamp}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.state.final_report)
        print(f"✅ Markdown report: {report_path}")
        
        # 2. JSON transcript
        json_path = output_dir / f"trial_transcript_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.state.transcript.dict(), f, indent=2, ensure_ascii=False)
        print(f"✅ JSON transcript: {json_path}")
        
        # 3. CSV table
        csv_path = output_dir / f"arguments_table_{timestamp}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Round,Side,Type,Content_Preview,Used_A2A,Timestamp\n")
            for arg in self.state.transcript.arguments:
                content_preview = arg.content.replace("\n", " ").replace('"', "'")[:100] + "..."
                f.write(f'{arg.round_number},"{arg.side}","{arg.argument_type}","{content_preview}",{arg.used_a2a},{arg.timestamp}\n')
        print(f"✅ Arguments table (CSV): {csv_path}")

# ===== Interactive Input Functions =====
def get_user_inputs():
    """Get user inputs for case selection and configuration."""
    from src.courtroom.tools.file_parser_tool import list_all_cases
    
    print("\n" + "="*80)
    print("🏛️  AI COURTROOM - FULL TRIAL SIMULATION")
    print("   All 6 Requirements Implemented:")
    print("   ✅ Context Sharing | ✅ MCP Tools | ✅ Structured Output")
    print("   ✅ Callbacks | ✅ A2A Protocol | ✅ CrewAI Framework")
    print("="*80)
    print(f"🤖 Using LLM: gemini-1.5-pro (separate quota pool)")
    print("="*80)
    
    # Quota warning
    used, limit = token_rate_limiter.get_daily_usage()
    if used > 0:
        print(f"\n⚠️  QUOTA WARNING: {used}/{limit} daily requests used today")
        print(f"   Gemini free tier limit: 250 requests/day")
        if used >= limit * 0.8:
            print(f"   ⚠️  WARNING: You're close to the daily limit!")
            print(f"   💡 Consider reducing argument rounds or waiting until tomorrow")
    print()
    
    # Get available cases
    print("📋 Loading available cases...")
    try:
        available_cases = list_all_cases()
        if not available_cases or "Error" in str(available_cases[0]):
            print("⚠️  Could not load cases. Using default case.")
            case_name = "Rajkumar"
        else:
            print(f"\n📚 Found {len(available_cases)} unique case(s):\n")
            for idx, case in enumerate(available_cases[:20], 1):  # Show first 20
                print(f"   {idx}. {case}")
            if len(available_cases) > 20:
                print(f"   ... and {len(available_cases) - 20} more cases")
            
            print("\n" + "-"*80)
            print("Case Selection:")
            print("-"*80)
            user_input = input("\nEnter case name (or number from list, or press Enter for default 'Rajkumar'): ").strip()
            
            if not user_input:
                case_name = "Rajkumar"
                print(f"✅ Using default case: {case_name}")
            elif user_input.isdigit():
                case_num = int(user_input)
                if 1 <= case_num <= len(available_cases):
                    case_name = available_cases[case_num - 1]
                    print(f"✅ Selected case: {case_name}")
                else:
                    print(f"⚠️  Invalid number. Using default case: Rajkumar")
                    case_name = "Rajkumar"
            else:
                case_name = user_input
                print(f"✅ Selected case: {case_name}")
    except Exception as e:
        print(f"⚠️  Error loading cases: {e}. Using default case.")
        case_name = "Rajkumar"
    
    # Get number of argument rounds
    print("\n" + "-"*80)
    print("Argument Rounds Configuration:")
    print("-"*80)
    print("   Recommended: 3-5 rounds (faster, ~50-80 API calls)")
    print("   Full trial: 10 rounds (more comprehensive, ~150-200 API calls)")
    print("\n   ⚠️  Note: Each round uses ~15-20 API calls")
    print(f"   📊 Current daily quota: {used}/{limit} requests used")
    
    if used >= limit * 0.7:
        print(f"   🚨 WARNING: High quota usage! Consider 3 rounds or wait until tomorrow")
    
    rounds_input = input("\nEnter number of argument rounds (3-10, default: 5): ").strip()
    
    if not rounds_input:
        num_rounds = 5
    else:
        try:
            num_rounds = int(rounds_input)
            if num_rounds < 3:
                print("⚠️  Minimum 3 rounds. Setting to 3.")
                num_rounds = 3
            elif num_rounds > 10:
                print("⚠️  Maximum 10 rounds. Setting to 10.")
                num_rounds = 10
        except ValueError:
            print("⚠️  Invalid input. Using default: 5 rounds")
            num_rounds = 5
    
    print(f"✅ Configured for {num_rounds} argument rounds")
    
    return {
        "case_name": case_name,
        "num_rounds": num_rounds
    }

# ===== Entry Point =====
def kickoff():
    """Main entry point with interactive user input."""
    # Get user inputs
    config = get_user_inputs()
    
    print("\n" + "="*80)
    print("🚀 Starting Trial Simulation...")
    print(f"   Case: {config['case_name']}")
    print(f"   Argument Rounds: {config['num_rounds']}")
    print("="*80 + "\n")
    
    # Create flow with user configuration
    flow = CourtroomFlow(case_name=config['case_name'], num_rounds=config['num_rounds'])
    flow.kickoff()

if __name__ == "__main__":
    kickoff()