from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, Request, BackgroundTasks, Response
from fastapi.responses import JSONResponse, Response as FastAPIResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import asyncio
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr, field_validator
import json
import re
import stripe
import requests as _requests
from typing import List, Optional, Dict, Any
import uuid
import secrets
from datetime import datetime, timezone, timedelta
import jwt
import bcrypt
from openai import AsyncOpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from astro_calculator import (
    calculate_natal_chart,
    calculate_current_transits,
    calculate_numerology,
    calculate_gematria,
    get_coordinates,
    get_element,
    get_modality
)

# ============== JSON Serialization Helpers ==============

def serialize_for_json(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable Python types.
    
    Handles numpy types, tuples, and other non-standard types that may
    appear in Swiss Ephemeris calculations.
    """
    import types
    
    # Handle None
    if obj is None:
        return None
    
    # Handle numpy types
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    if hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    
    # Handle tuple - convert to list
    if isinstance(obj, tuple):
        return [serialize_for_json(item) for item in obj]
    
    # Handle types.SimpleNamespace (like namespace from functools)
    if isinstance(obj, types.SimpleNamespace):
        return serialize_for_json(vars(obj))
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {str(k): serialize_for_json(v) for k, v in obj.items()}
    
    # Handle lists/tuples
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [serialize_for_json(item) for item in obj]
    
    # Handle numbers (int, float, numpy number)
    if isinstance(obj, (int, float)):
        # Check for numpy number types
        if hasattr(obj, 'dtype'):
            return float(obj) if 'float' in str(obj.dtype) else int(obj)
        return obj
    
    # Handle booleans
    if isinstance(obj, bool):
        return obj
    
    # Handle strings
    if isinstance(obj, str):
        return obj
    
    # Handle datetime objects
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # For any other type, try to convert to string
    try:
        return str(obj)
    except:
        return None


def validate_chart_response(chart_data: dict) -> bool:
    """Validate that chart data has all required fields and proper types."""
    required_string_fields = ['sun_sign', 'moon_sign', 'rising_sign']
    required_dict_fields = ['planets', 'houses', 'aspects']
    
    for field in required_string_fields:
        if field not in chart_data or not isinstance(chart_data[field], str):
            logging.warning(f"Chart validation failed: missing or invalid {field}")
            return False
    
    for field in required_dict_fields:
        if field not in chart_data or not isinstance(chart_data[field], (dict, list)):
            logging.warning(f"Chart validation failed: missing or invalid {field}")
            return False
    
    return True



# Modular engines (from Gab44-vision merge)
from numerology import calculate_full_profile as numerology_full_profile
from gematria import calculate_all as gematria_calculate_all
from cities import geocode_search
from chart_image import render_wheel, render_share_card, to_png_bytes

ROOT_DIR = Path(__file__).parent
load_dotenv('/root/secrets/all-keys.env')

# ── Startup environment validation ───────────────────────────────────────────
_REQUIRED_ENV_VARS = {
    'MONGO_URL': (
        'MongoDB connection string. '
        'On Railway: add a MongoDB database to your project and Railway will '
        'inject this automatically, OR set it manually in the service variables.'
    ),
    'DB_NAME': (
        'MongoDB database name (e.g. "gab44"). '
        'Set this in the Railway service variables.'
    ),
    'JWT_SECRET': (
        "Long random secret for JWT signing. "
        "Generate one with: python -c 'import secrets; print(secrets.token_hex(32))'"
    ),
}
_missing_vars = [var for var in _REQUIRED_ENV_VARS if not os.environ.get(var)]
if _missing_vars:
    lines = [
        '',
        '=' * 70,
        'STARTUP ERROR — Missing required environment variables:',
        '',
    ]
    for var in _missing_vars:
        lines.append(f'  {var}')
        lines.append(f'    → {_REQUIRED_ENV_VARS[var]}')
        lines.append('')
    lines += [
        'Set these variables in your Railway service dashboard before deploying.',
        'See the README "Railway Deployment" section for the full setup guide.',
        '=' * 70,
        '',
    ]
    raise EnvironmentError('\n'.join(lines))

# MongoDB connection (bypassed due to service unavailability)
# mongo_url = os.environ['MONGO_URL']
# client = AsyncIOMotorClient(mongo_url, maxPoolSize=50, minPoolSize=5)
# db = client[os.environ['DB_NAME']]

# JWT Configuration
JWT_SECRET = os.environ['JWT_SECRET']
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # 7 days

# OpenAI / LLM Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ElevenLabs TTS Configuration (voice horoscope feature)
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY', '')
ELEVENLABS_VOICE_ID = os.environ.get('ELEVENLABS_VOICE_ID', 'EXAVITQu4vr4xnSDxMaL')  # "Sarah" - warm female narrator
ELEVENLABS_MODEL_ID = os.environ.get('ELEVENLABS_MODEL_ID', 'eleven_turbo_v2_5')
VOICE_HOROSCOPE_TIERS = {"enthusiast", "advanced", "professional"}


# Admin emails (set via environment variable)
ADMIN_EMAILS = [e.strip().lower() for e in os.environ.get('ADMIN_EMAILS', '').split(',') if e.strip()]

# Stripe Configuration
STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET', '')
STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY', '')
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# Plan → price (in cents) mapping for inline checkout
STRIPE_PLAN_PRICES = {
    "enthusiast": {"amount": 1999, "name": "Gab44 Enthusiast"},
    "advanced":   {"amount": 4999, "name": "Gab44 Advanced"},
    "professional": {"amount": 9900, "name": "Gab44 Professional"},
}

# One-time products (mode=payment). Single source of truth for SKUs.
PERSONAL_READING_SKU = "personal_reading"
PERSONAL_READING_PRICE_CENTS = 1900
PERSONAL_READING_NAME = "Gab44 Personal Astrology Reading"
PERSONAL_READING_DESCRIPTION = (
    "A one-time, fully personalized astrology reading drawn from your full natal chart. "
    "Delivered as a written report within 48 hours."
)

# OneSignal Configuration
ONESIGNAL_APP_ID = os.environ.get('ONESIGNAL_APP_ID', '')
ONESIGNAL_API_KEY = os.environ.get('ONESIGNAL_API_KEY', '')
ONESIGNAL_API_URL = "https://api.onesignal.com/notifications"

# SendGrid Configuration
SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY', '')

# Separate sender addresses per email type (all must be verified in SendGrid)
EMAIL_NOREPLY   = os.environ.get('EMAIL_NOREPLY',   'noreply@gab44.com')    # system / automated
EMAIL_VERIFY    = os.environ.get('EMAIL_VERIFY',    'verify@gab44.com')     # sign-up & verification
EMAIL_SUPPORT   = os.environ.get('EMAIL_SUPPORT',   'support@gab44.com')    # support replies
EMAIL_MARKETING = os.environ.get('EMAIL_MARKETING', 'hello@gab44.com')      # promotions & newsletters

# Token validity windows
EMAIL_VERIFY_EXPIRY_HOURS = 48
PASSWORD_RESET_EXPIRY_HOURS = 1

# Birth chart cache schema version — increment when the planets structure changes
# so that stale cached documents are automatically refreshed.
BIRTH_CHART_SCHEMA_VERSION = 1

# Chat message limits per subscription tier (per calendar day, UTC)
CHAT_DAILY_LIMITS: dict[str, int] = {
    "seeker":       10,
    "enthusiast":   100,
    "advanced":     -1,   # unlimited
    "professional": -1,   # unlimited
}

app = FastAPI(title="Gab44 - Astrology AI Coaching Platform")
api_router = APIRouter(prefix="/api")
# Temporarily disabled for segmentation fault investigation
# from voice_horoscope import router as voice_horoscope_router
# app.include_router(voice_horoscope_router)
# Temporarily disabled due to segmentation fault investigation
# from birth_chart_image import router as birth_chart_router
# app.include_router(birth_chart_router)
from zodiac_seo import router as zodiac_seo_router
app.include_router(zodiac_seo_router)
from conversion_optimization import router as conversion_router
app.include_router(conversion_router)
from mobile_optimization import router as mobile_router
app.include_router(mobile_router)
from performance_optimization import router as performance_router
app.include_router(performance_router)
from conversion_optimization import router as conversion_router
app.include_router(conversion_router)

# Temporary voice horoscope endpoint (bypasses disabled router segmentation fault)
@api_router.get('/voice-horoscope/{readingId}')
@track_endpoint_performance
async def get_voice_horoscope(readingId: str, current_user: dict = Depends(get_optional_user)):
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="Voice feature unavailable")
    # Dummy horoscope text (replace with actual reading fetch when MongoDB is available)
    dummy_horoscope = f"Premium voice horoscope for reading {readingId}: Your stars align for new opportunities this week."
    try:
        client = elevenlabs.ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio = await client.textToSpeech.convert(
            voiceId=ELEVENLABS_VOICE_ID,
            outputFormat='mp3',
            text=dummy_horoscope,
            modelId=ELEVENLABS_MODEL_ID
        )
        return {'audioUrl': f'data:audio/mpeg;base64,{elevenlabs.to_base64(audio)}'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice generation failed: {str(e)}")

app.include_router(api_router)
security = HTTPBearer()

# Rate limiter (uses client IP address)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============== Security Headers Middleware ==============

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security-related HTTP response headers to every response."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

# ============== Models ==============

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str
    birth_name: Optional[str] = None  # Legal birth name for numerology (if different from display name)
    birth_date: str  # YYYY-MM-DD
    birth_time: Optional[str] = None  # HH:MM
    birth_place: str
    birth_latitude: Optional[float] = None
    birth_longitude: Optional[float] = None

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Za-z]", v):
            raise ValueError("Password must contain at least one letter")
        if not re.search(r"[0-9!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain at least one digit or special character")
        return v


class UserUpdate(BaseModel):
    name: Optional[str] = None
    birth_name: Optional[str] = None
    birth_date: Optional[str] = None
    birth_time: Optional[str] = None
    birth_place: Optional[str] = None
    birth_latitude: Optional[float] = None
    birth_longitude: Optional[float] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r"[A-Za-z]", v):
            raise ValueError("Password must contain at least one letter")
        if not re.search(r"[0-9!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain at least one digit or special character")
        return v

class UserProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    email: str
    name: str
    birth_name: Optional[str] = None
    birth_date: str
    birth_time: Optional[str] = None
    birth_place: str
    birth_latitude: Optional[float] = None
    birth_longitude: Optional[float] = None
    sun_sign: Optional[str] = None
    subscription_tier: str = "seeker"
    is_admin: bool = False
    email_verified: bool = False
    created_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfile

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

class BirthChart(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    user_id: str
    sun_sign: str
    moon_sign: str
    rising_sign: str
    planets: dict
    houses: dict
    aspects: List[dict]
    patterns: List[str]
    created_at: str

class TransitActivation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    user_id: str
    transit_type: str
    planet: str
    aspect: str
    natal_planet: str
    start_date: str
    peak_date: str
    end_date: str
    strength: float
    interpretation: str
    action_items: List[str]

class DailyGuidance(BaseModel):
    date: str
    overall_energy: str
    focus_areas: List[str]
    action_items: List[str]
    transit_highlights: List[dict]

# ============== Compatibility Models ==============

class CompatibilityRequest(BaseModel):
    partner_name: str
    partner_birth_date: str  # YYYY-MM-DD
    partner_birth_time: Optional[str] = None
    partner_birth_place: str
    partner_birth_name: Optional[str] = None  # legal birth name for numerology
    relationship_type: str = "romantic"  # romantic | friendship | family | business | colleague

    @validator('partner_birth_time')
    def validate_birth_time(cls, v):
        # Empty string is not allowed - convert to None
        if v == '':
            return None
        return v

    @validator('partner_birth_place')
    def validate_birth_place(cls, v):
        # Validate that place is not empty and not the default ocean coordinates
        if not v or not v.strip():
            raise ValueError('Birth place is required')
        # Check for invalid coordinates (0.0, 0.0) - this is in the Atlantic Ocean
        v_lower = v.lower().strip()
        if v_lower in ['0,0', '0.0, 0.0', '0 0', '(0, 0)', '(0.0, 0.0)']:
            raise ValueError('Please provide a valid city or location')
        return v

    @validator('partner_birth_date')
    def validate_birth_date(cls, v):
        # Validate date format YYYY-MM-DD
        from datetime import datetime
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Birth date must be in YYYY-MM-DD format')
        return v

class SynastryAspect(BaseModel):
    person1_planet: str
    person2_planet: str
    aspect_type: str
    orb: float
    strength: float
    interpretation: str
    category: str  # romantic, communication, emotional, karmic, growth

class CompatibilityReport(BaseModel):
    id: str
    user_id: str
    partner_name: str
    partner_birth_date: str
    partner_sun_sign: str
    overall_score: float
    category_scores: Dict[str, float]
    synastry_aspects: List[Dict[str, Any]]
    composite_chart: Dict[str, Any]
    strengths: List[str]
    challenges: List[str]
    karmic_themes: List[str]
    growth_opportunities: List[str]
    ai_analysis: str
    created_at: str

# ============== Payment + Notification Models ==============

class CheckoutRequest(BaseModel):
    tier: str  # enthusiast | advanced | professional

class BuyReadingRequest(BaseModel):
    """One-time $19 personal reading. Email is required for guest checkout
    (Stripe will also collect/verify it); birth fields are optional context
    the buyer can include up-front so we have a head start when delivering."""
    email: Optional[EmailStr] = None
    name: Optional[str] = None
    birth_date: Optional[str] = None
    birth_time: Optional[str] = None
    birth_place: Optional[str] = None
    notes: Optional[str] = Field(default=None, max_length=2000)

class DeviceRegistration(BaseModel):
    player_id: str  # OneSignal player/subscription ID

class NewsletterSubscription(BaseModel):
    email: EmailStr
    name: str = ""

class ContactMessage(BaseModel):
    name: str
    email: EmailStr
    subject: str = Field(..., max_length=200)
    message: str = Field(..., min_length=10, max_length=5000)

class EmailBlastRequest(BaseModel):
    subject: str
    body_html: str
    tier_filter: str = "all"  # "all" | "seeker" | "enthusiast" | "advanced" | "professional"

# ============== Auth Helpers ==============

# Fields that must never appear in API responses
_SENSITIVE_USER_FIELDS = frozenset({
    "_id", "password",
    "email_verification_token",
    "password_reset_token", "password_reset_expiry",
})

# Pre-computed dummy hash used during login when no user is found, so that the
# bcrypt comparison cost is always paid and response time stays constant
# (prevents user-enumeration via timing differences).
_DUMMY_HASH: str = bcrypt.hashpw(b"gab44_timing_resist_dummy", bcrypt.gensalt()).decode()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_token(user_id: str) -> str:
    expiration = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {"user_id": user_id, "exp": expiration}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = await db.users.find_one(
            {"id": user_id},
            {"_id": 0, "password": 0,
             "email_verification_token": 0,
             "password_reset_token": 0, "password_reset_expiry": 0}
        )
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        # Admin status from DB role OR from ADMIN_EMAILS env var (bootstrap)
        user_email = user.get("email", "").lower()
        is_admin_by_role = user.get("role") == "admin"
        is_admin_by_env = user_email in ADMIN_EMAILS
        user["is_admin"] = is_admin_by_role or is_admin_by_env
        
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Dependency that requires admin access"""
    if not user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

async def get_optional_user(request: Request) -> Optional[dict]:
    """Extract a user from the Authorization header if present, else None.
    Never raises — used by endpoints that must work for both authed and guest
    callers (e.g. one-time-purchase checkout)."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return None
    token = auth.split(" ", 1)[1].strip()
    if not token:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if not user_id:
            return None
        user = await db.users.find_one(
            {"id": user_id},
            {"_id": 0, "password": 0,
             "email_verification_token": 0,
             "password_reset_token": 0, "password_reset_expiry": 0}
        )
        return user
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

# ============== Email Helper (SendGrid) ==============

def send_email(to_email: str, subject: str, html_content: str, from_email: str = None) -> bool:
    """Send an email via SendGrid.
    
    Pass ``from_email`` to override the default sender (EMAIL_NOREPLY).
    Use the purpose-specific constants:
      EMAIL_VERIFY    – account / sign-up emails
      EMAIL_SUPPORT   – support replies
      EMAIL_MARKETING – promotions & newsletters
      EMAIL_NOREPLY   – automated system emails (default)
    """
    if not SENDGRID_API_KEY:
        logging.warning("SENDGRID_API_KEY not configured – email not sent to %s", to_email)
        return False
    sender = from_email or EMAIL_NOREPLY
    try:
        message = Mail(
            from_email=sender,
            to_emails=to_email,
            subject=subject,
            html_content=html_content,
        )
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        logging.info("Email sent from %s to %s – status %s", sender, to_email, response.status_code)
        return response.status_code in (200, 202)
    except Exception as exc:
        logging.error("SendGrid error sending to %s: %s", to_email, exc)
        return False

# ---------- Email template builders ----------

_EMAIL_BASE = """
<div style="font-family:sans-serif;max-width:520px;margin:auto;padding:32px 24px;
            background:#0a0a0f;color:#e8e0f0;border-radius:16px;">
  {body}
  <hr style="border:none;border-top:1px solid #222;margin:24px 0;" />
  <p style="color:#606080;font-size:12px;margin:0;">
    Gab44 &middot; Astrology AI Coaching &middot;
    <a href="mailto:{support}" style="color:#c9a84c;">{support}</a>
  </p>
</div>
"""

def _email_wrap(body: str) -> str:
    return _EMAIL_BASE.format(body=body, support=EMAIL_SUPPORT)


def build_verification_email(name: str, verify_url: str) -> str:
    """Sign-up verification email — sent from EMAIL_VERIFY."""
    body = f"""
      <h1 style="font-family:Georgia,serif;font-size:28px;margin-bottom:4px;color:#c9a84c;">
        ✨ Welcome to Gab44</h1>
      <p style="color:#9090b0;margin-top:0;">Your cosmic journey starts with one click.</p>
      <p style="margin-top:24px;">Hi {name},</p>
      <p>Please verify your email address so we can keep your account secure and
         send you personalized astrological guidance.</p>
      <a href="{verify_url}"
         style="display:inline-block;margin:24px 0;padding:14px 32px;background:#c9a84c;
                color:#0a0a0f;font-weight:700;border-radius:12px;text-decoration:none;font-size:16px;">
        Verify My Email
      </a>
      <p style="color:#9090b0;font-size:13px;">
        This link expires in {EMAIL_VERIFY_EXPIRY_HOURS}&nbsp;hours. If you didn't create a Gab44 account,
        you can safely ignore this email.</p>
    """
    return _email_wrap(body)


def build_welcome_email(name: str, dashboard_url: str) -> str:
    """Post-verification welcome email — sent from EMAIL_VERIFY."""
    body = f"""
      <h1 style="font-family:Georgia,serif;font-size:28px;margin-bottom:4px;color:#c9a84c;">
        🌟 You're verified, {name}!</h1>
      <p style="color:#9090b0;margin-top:0;">The cosmos is ready to guide you.</p>
      <p style="margin-top:24px;">Your Gab44 account is now fully activated.</p>
      <p>Head to your dashboard to explore your natal chart, daily guidance, and AI coaching.</p>
      <a href="{dashboard_url}"
         style="display:inline-block;margin:24px 0;padding:14px 32px;background:#c9a84c;
                color:#0a0a0f;font-weight:700;border-radius:12px;text-decoration:none;font-size:16px;">
        Go to My Dashboard
      </a>
    """
    return _email_wrap(body)


def build_promotional_email(name: str, headline: str, body_html: str, cta_text: str, cta_url: str) -> str:
    """Generic promotional / newsletter email — sent from EMAIL_MARKETING."""
    body = f"""
      <h1 style="font-family:Georgia,serif;font-size:26px;margin-bottom:4px;color:#c9a84c;">
        {headline}</h1>
      <p style="margin-top:16px;">Hi {name},</p>
      {body_html}
      <a href="{cta_url}"
         style="display:inline-block;margin:24px 0;padding:14px 32px;background:#c9a84c;
                color:#0a0a0f;font-weight:700;border-radius:12px;text-decoration:none;font-size:16px;">
        {cta_text}
      </a>
      <p style="color:#9090b0;font-size:12px;">
        You're receiving this because you signed up for Gab44 updates.
        To unsubscribe, reply with "unsubscribe" to
        <a href="mailto:{EMAIL_MARKETING}" style="color:#c9a84c;">{EMAIL_MARKETING}</a>.
      </p>
    """
    return _email_wrap(body)


def build_support_reply_email(name: str, ticket_subject: str, reply_body: str) -> str:
    """Support reply email — sent from EMAIL_SUPPORT."""
    body = f"""
      <h1 style="font-family:Georgia,serif;font-size:22px;margin-bottom:4px;color:#c9a84c;">
        Re: {ticket_subject}</h1>
      <p style="margin-top:16px;">Hi {name},</p>
      {reply_body}
      <p style="margin-top:24px;">Need more help?
        <a href="mailto:{EMAIL_SUPPORT}" style="color:#c9a84c;">Reply to this email</a>
        and we'll get back to you.</p>
    """
    return _email_wrap(body)


def build_password_reset_email(name: str, reset_url: str) -> str:
    """Password-reset email — sent from EMAIL_NOREPLY."""
    body = f"""
      <h1 style="font-family:Georgia,serif;font-size:26px;margin-bottom:4px;color:#c9a84c;">
        🔑 Reset your password</h1>
      <p style="color:#9090b0;margin-top:0;">A reset was requested for your Gab44 account.</p>
      <p style="margin-top:24px;">Hi {name},</p>
      <p>Click the button below to choose a new password. This link expires in
         <strong>{PASSWORD_RESET_EXPIRY_HOURS}&nbsp;hour</strong>.</p>
      <a href="{reset_url}"
         style="display:inline-block;margin:24px 0;padding:14px 32px;background:#c9a84c;
                color:#0a0a0f;font-weight:700;border-radius:12px;text-decoration:none;font-size:16px;">
        Reset My Password
      </a>
      <p style="color:#9090b0;font-size:13px;">
        If you didn't request a password reset, you can safely ignore this email —
        your password will not change.</p>
    """
    return _email_wrap(body)


# ============== Astrology Helpers ==============

def calculate_sun_sign(birth_date: str) -> str:
    """Calculate sun sign using Swiss Ephemeris for exact accuracy (handles cusp days correctly)."""
    try:
        chart = calculate_natal_chart(birth_date)
        return chart.get("sun_sign", "Unknown")
    except Exception:
        return "Unknown"

def get_sign_element(sign: str) -> str:
    """Get the element for a zodiac sign"""
    elements = {
        "Aries": "Fire", "Leo": "Fire", "Sagittarius": "Fire",
        "Taurus": "Earth", "Virgo": "Earth", "Capricorn": "Earth",
        "Gemini": "Air", "Libra": "Air", "Aquarius": "Air",
        "Cancer": "Water", "Scorpio": "Water", "Pisces": "Water"
    }
    return elements.get(sign, "Unknown")

def get_sign_modality(sign: str) -> str:
    """Get the modality for a zodiac sign"""
    modalities = {
        "Aries": "Cardinal", "Cancer": "Cardinal", "Libra": "Cardinal", "Capricorn": "Cardinal",
        "Taurus": "Fixed", "Leo": "Fixed", "Scorpio": "Fixed", "Aquarius": "Fixed",
        "Gemini": "Mutable", "Virgo": "Mutable", "Sagittarius": "Mutable", "Pisces": "Mutable"
    }
    return modalities.get(sign, "Unknown")

def get_sign_polarity(sign: str) -> str:
    """Get the polarity (Yin/Yang) for a zodiac sign"""
    yang_signs = ["Aries", "Gemini", "Leo", "Libra", "Sagittarius", "Aquarius"]
    return "Yang" if sign in yang_signs else "Yin"

def get_ruling_planet(sign: str) -> str:
    """Get the ruling planet for a zodiac sign"""
    rulers = {
        "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury", "Cancer": "Moon",
        "Leo": "Sun", "Virgo": "Mercury", "Libra": "Venus", "Scorpio": "Pluto",
        "Sagittarius": "Jupiter", "Capricorn": "Saturn", "Aquarius": "Uranus", "Pisces": "Neptune"
    }
    return rulers.get(sign, "Unknown")

# ============== Compatibility Calculations ==============

def calculate_element_compatibility(sign1: str, sign2: str) -> dict:
    """Calculate compatibility based on elements"""
    elem1 = get_sign_element(sign1)
    elem2 = get_sign_element(sign2)
    
    # Same element = high compatibility
    if elem1 == elem2:
        return {"score": 95, "description": f"Both {elem1} signs - deep natural understanding and shared values"}
    
    # Compatible elements
    compatible_pairs = {
        ("Fire", "Air"): {"score": 85, "description": "Fire and Air fuel each other - exciting and stimulating"},
        ("Earth", "Water"): {"score": 85, "description": "Earth and Water nurture each other - stable and emotionally supportive"},
    }
    
    pair = tuple(sorted([elem1, elem2]))
    if pair in compatible_pairs:
        return compatible_pairs[pair]
    if (elem2, elem1) in compatible_pairs:
        return compatible_pairs[(elem2, elem1)]
    
    # Challenging but growth-oriented
    challenging_pairs = {
        ("Fire", "Water"): {"score": 55, "description": "Fire and Water create steam - passionate but requires balance"},
        ("Fire", "Earth"): {"score": 60, "description": "Fire and Earth - different paces, but can ground and inspire each other"},
        ("Air", "Water"): {"score": 58, "description": "Air and Water - mind vs heart, requires conscious communication"},
        ("Air", "Earth"): {"score": 62, "description": "Air and Earth - ideas vs practicality, complementary if respected"},
    }
    
    pair = tuple(sorted([elem1, elem2]))
    if pair in challenging_pairs:
        return challenging_pairs[pair]
    if (elem2, elem1) in challenging_pairs:
        return challenging_pairs[(elem2, elem1)]
    
    return {"score": 65, "description": "Neutral elemental compatibility"}

def calculate_modality_compatibility(sign1: str, sign2: str) -> dict:
    """Calculate compatibility based on modalities"""
    mod1 = get_sign_modality(sign1)
    mod2 = get_sign_modality(sign2)
    
    if mod1 == mod2:
        if mod1 == "Fixed":
            return {"score": 70, "description": "Both Fixed - loyal and stable, but may struggle with change"}
        elif mod1 == "Cardinal":
            return {"score": 72, "description": "Both Cardinal - dynamic leaders, may compete for direction"}
        else:
            return {"score": 78, "description": "Both Mutable - adaptable and flexible together"}
    
    # Mixed modalities
    if "Cardinal" in [mod1, mod2] and "Fixed" in [mod1, mod2]:
        return {"score": 75, "description": "Cardinal-Fixed: Initiative meets stability"}
    if "Cardinal" in [mod1, mod2] and "Mutable" in [mod1, mod2]:
        return {"score": 82, "description": "Cardinal-Mutable: Leadership meets adaptability"}
    if "Fixed" in [mod1, mod2] and "Mutable" in [mod1, mod2]:
        return {"score": 80, "description": "Fixed-Mutable: Stability meets flexibility"}
    
    return {"score": 75, "description": "Balanced modality interaction"}

def calculate_synastry_aspects(chart1: dict, chart2: dict) -> list:
    """Calculate synastry aspects between two charts"""
    aspects = []
    aspect_types = {
        0: {"name": "conjunction", "harmony": 0.9, "category": "intense"},
        60: {"name": "sextile", "harmony": 0.8, "category": "harmonious"},
        90: {"name": "square", "harmony": 0.4, "category": "challenging"},
        120: {"name": "trine", "harmony": 0.95, "category": "harmonious"},
        180: {"name": "opposition", "harmony": 0.5, "category": "polarizing"}
    }
    
    planets = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn"]
    
    for p1 in planets:
        for p2 in planets:
            if p1 in chart1.get("planets", {}) and p2 in chart2.get("planets", {}):
                deg1 = chart1["planets"][p1].get("degree", 0)
                deg2 = chart2["planets"][p2].get("degree", 0)
                
                # Calculate angular difference
                diff = abs(deg1 - deg2) % 360
                if diff > 180:
                    diff = 360 - diff
                
                # Check for aspects with orb
                for angle, aspect_info in aspect_types.items():
                    orb = abs(diff - angle)
                    if orb <= 8:  # 8 degree orb
                        strength = 1 - (orb / 8)
                        
                        # Categorize the aspect
                        category = "general"
                        if p1 in ["venus", "mars"] or p2 in ["venus", "mars"]:
                            category = "romantic"
                        elif p1 == "moon" or p2 == "moon":
                            category = "emotional"
                        elif p1 in ["mercury"] or p2 in ["mercury"]:
                            category = "communication"
                        elif p1 in ["saturn", "pluto"] or p2 in ["saturn", "pluto"]:
                            category = "karmic"
                        elif p1 == "jupiter" or p2 == "jupiter":
                            category = "growth"
                        
                        aspects.append({
                            "person1_planet": p1,
                            "person2_planet": p2,
                            "aspect_type": aspect_info["name"],
                            "orb": round(orb, 2),
                            "strength": round(strength, 2),
                            "harmony": aspect_info["harmony"],
                            "category": category
                        })
    
    return sorted(aspects, key=lambda x: x["strength"], reverse=True)[:15]

def generate_composite_chart(chart1: dict, chart2: dict) -> dict:
    """Generate a composite chart from two birth charts"""
    composite = {"planets": {}, "houses": {}}
    
    planets = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn"]
    
    for planet in planets:
        if planet in chart1.get("planets", {}) and planet in chart2.get("planets", {}):
            deg1 = chart1["planets"][planet].get("degree", 0)
            deg2 = chart2["planets"][planet].get("degree", 0)
            
            # Calculate midpoint
            midpoint = (deg1 + deg2) / 2
            if abs(deg1 - deg2) > 180:
                midpoint = (midpoint + 180) % 360
            
            # Determine sign from degree
            sign_index = int(midpoint / 30)
            signs = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
                    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
            
            composite["planets"][planet] = {
                "degree": round(midpoint % 30, 2),
                "sign": signs[sign_index % 12],
                "house": chart1["planets"][planet].get("house", 1)
            }
    
    return composite

# Per-type AI framing config
_REL_TYPE_CONFIG = {
    "romantic": {
        "system": "You are Gab44, an expert relationship astrologer specialising in romantic compatibility. Provide insightful, compassionate analysis that illuminates the love potential, passion, and long-term romantic growth of this pairing.",
        "context": "ROMANTIC PARTNERSHIP",
        "question_5": "Practical advice for deepening intimacy and building a lasting romantic bond",
    },
    "friendship": {
        "system": "You are Gab44, an expert astrologer specialising in friendship and platonic compatibility. Provide warm, insightful analysis of how these two people can build and sustain a meaningful, supportive friendship.",
        "context": "FRIENDSHIP",
        "question_5": "Practical advice for deepening trust and creating a lasting, joyful friendship",
    },
    "family": {
        "system": "You are Gab44, an expert astrologer specialising in family and ancestral dynamics. Provide compassionate, grounded analysis of how these two people can understand their family bond, heal patterns, and support each other.",
        "context": "FAMILY RELATIONSHIP",
        "question_5": "Practical advice for honouring the family bond and healing any ancestral patterns",
    },
    "business": {
        "system": "You are Gab44, an expert astrologer specialising in business partnerships and professional compatibility. Provide strategic, insightful analysis of how these two people can build a successful, aligned business partnership.",
        "context": "BUSINESS PARTNERSHIP",
        "question_5": "Practical advice for building a successful, long-term business collaboration",
    },
    "colleague": {
        "system": "You are Gab44, an expert astrologer specialising in professional and collegial dynamics. Provide practical, insightful analysis of how these two people can work effectively together and leverage each other's strengths.",
        "context": "PROFESSIONAL / COLLEAGUE RELATIONSHIP",
        "question_5": "Practical advice for a productive, harmonious working relationship",
    },
}

async def generate_compatibility_analysis(user: dict, partner_data: dict, synastry: list, scores: dict) -> str:
    """Generate AI-powered compatibility analysis (any relationship type)"""

    relationship_type = partner_data.get("relationship_type", "romantic")
    cfg = _REL_TYPE_CONFIG.get(relationship_type, _REL_TYPE_CONFIG["romantic"])
    
    user_sun = user.get("sun_sign", "Unknown")
    partner_sun = partner_data.get("sun_sign", "Unknown")
    
    # Format synastry aspects for analysis
    aspect_summary = "\n".join([
        f"- {a['person1_planet'].title()} {a['aspect_type']} {a['person2_planet'].title()} ({a['category']})"
        for a in synastry[:8]
    ])

    # Numerology comparison block (included when available)
    num_block = ""
    user_num = partner_data.get("user_numerology", {})
    partner_num = partner_data.get("partner_numerology", {})
    if user_num and partner_num:
        u_lp = user_num.get("life_path", {}).get("number", "?")
        p_lp = partner_num.get("life_path", {}).get("number", "?")
        u_su = user_num.get("soul_urge", {}).get("number", "?")
        p_su = partner_num.get("soul_urge", {}).get("number", "?")
        u_ex = user_num.get("expression", {}).get("number", "?")
        p_ex = partner_num.get("expression", {}).get("number", "?")
        num_block = f"""
NUMEROLOGY COMPARISON:
- Life Path: {user.get('name', 'Person 1')} = {u_lp}  |  {partner_data.get('name', 'Person 2')} = {p_lp}{"  (MATCH!)" if str(u_lp) == str(p_lp) else ""}
- Soul Urge: {u_su}  |  {p_su}{"  (MATCH!)" if str(u_su) == str(p_su) else ""}
- Expression: {u_ex}  |  {p_ex}{"  (MATCH!)" if str(u_ex) == str(p_ex) else ""}"""

    connection_label = "Romance" if relationship_type == "romantic" else "Connection"

    prompt = f"""Provide a comprehensive {cfg['context']} compatibility analysis:

PERSON 1: {user.get('name')} - {user_sun} Sun
PERSON 2: {partner_data.get('name')} - {partner_sun} Sun
RELATIONSHIP TYPE: {cfg['context']}

COMPATIBILITY SCORES:
- Overall: {scores.get('overall', 0)}%
- Emotional: {scores.get('emotional', 0)}%
- Communication: {scores.get('communication', 0)}%
- {connection_label}: {scores.get('romantic', 0)}%
- Long-term Stability: {scores.get('stability', 0)}%
- Karmic Bond: {scores.get('karmic', 0)}%

KEY SYNASTRY ASPECTS:
{aspect_summary}
{num_block}
Provide:
1. Overall dynamic and natural chemistry for this {cfg['context'].lower()}
2. Key strengths of this pairing
3. Potential challenges and how to navigate them
4. Karmic or spiritual themes in this connection (include numerology Life Path resonance if provided)
5. {cfg['question_5']}

Keep the analysis warm, insightful, and actionable. Frame everything specifically for a {cfg['context'].lower()} — avoid defaulting to romantic language unless this is a romantic relationship."""

    try:
        if not openai_client:
            raise RuntimeError("OpenAI not configured")
        completion = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": cfg["system"]},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Compatibility analysis error: {e}")
        return f"Based on the {user_sun}-{partner_sun} pairing, this {relationship_type} connection shows {scores.get('overall', 70)}% compatibility. Key themes include balancing {get_sign_element(user_sun)} and {get_sign_element(partner_sun)} energies."

# ============== AI Coach ==============

async def get_ai_coach_response(user: dict, message: str, session_id: str) -> str:
    """Generate AI coaching response"""

    # Get user's chat history
    history = await db.chat_messages.find(
        {"user_id": user["id"], "session_id": session_id},
        {"_id": 0}
    ).sort("timestamp", 1).limit(20).to_list(20)

    # Build context
    sun_sign = user.get("sun_sign", "Unknown")
    element  = get_sign_element(sun_sign)
    modality = get_sign_modality(sun_sign)

    # Fetch full natal chart (cached in DB)
    chart_doc = await db.birth_charts.find_one({"user_id": user["id"]}, {"_id": 0})

    # Build planet summary
    planet_lines = []
    if chart_doc and chart_doc.get("planets"):
        for pname, pdata in chart_doc["planets"].items():
            if pdata:
                h = f" H{pdata['house']}" if pdata.get("house") else ""
                r = " ℞" if pdata.get("retrograde") else ""
                planet_lines.append(f"  {pname.replace('_',' ').title()}: {pdata.get('sign','?')} {pdata.get('sign_degree',0):.1f}°{h}{r}")
    planets_block = "\n".join(planet_lines) if planet_lines else "  (chart not yet generated)"

    # Build numerology summary
    num_lines = []
    if chart_doc and chart_doc.get("numerology"):
        num = chart_doc["numerology"]
        for key in ("life_path", "expression", "soul_urge", "personality", "personal_year"):
            entry = num.get(key, {})
            if entry:
                num_lines.append(f"  {key.replace('_',' ').title()}: {entry['number']} ({entry.get('keyword','')})")
    numerology_block = "\n".join(num_lines) if num_lines else "  (not calculated)"

    # Top aspects
    aspect_lines = []
    if chart_doc and chart_doc.get("aspects"):
        for a in chart_doc["aspects"][:5]:
            aspect_lines.append(f"  {a['planet1'].title()} {a['aspect']} {a['planet2'].title()} (orb {a['orb']}°)")
    aspects_block = "\n".join(aspect_lines) if aspect_lines else "  (none)"

    rising = chart_doc.get("rising_sign", "Unknown") if chart_doc else "Unknown"
    moon   = chart_doc.get("moon_sign",   "Unknown") if chart_doc else "Unknown"

    system_message = f"""You are Gab44, an advanced astrology AI coach. Your mission is to help people live measurably better lives through truthful astrological guidance.

USER PROFILE:
- Name: {user.get('name', 'Seeker')}
- Sun Sign: {sun_sign} ({element}, {modality})
- Moon Sign: {moon}
- Rising Sign: {rising}
- Birth Date: {user.get('birth_date', 'Unknown')}
- Birth Place: {user.get('birth_place', 'Unknown')}

NATAL PLANETS:
{planets_block}

TOP ASPECTS:
{aspects_block}

NUMEROLOGY:
{numerology_block}

YOUR PRINCIPLES:
1. Every response must help the user make better decisions
2. Be truthful, even when uncomfortable — truth serves growth
3. Provide actionable guidance, not just information
4. Weave together astrology AND numerology when they reinforce each other
5. Connect insights to practical life outcomes
6. Ask follow-up questions to understand if guidance was helpful

RESPONSE FORMAT:
- Keep responses conversational but substantive
- Include specific action items when relevant
- Reference planets, houses, and numerology numbers when applicable
- End with a thoughtful question or call to action"""

    try:
        if not openai_client:
            raise RuntimeError("OpenAI not configured")
        messages = [{"role": "system", "content": system_message}]
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        completion = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        return completion.choices[0].message.content

    except Exception as e:
        logging.error(f"AI Coach error: {e}")
        return f"I sense a disturbance in the cosmic connection. Let me try again... As a {sun_sign}, your {element} energy suggests taking a moment to ground yourself. What specific area of life would you like guidance on?"


async def get_ai_friend_response(user: dict, message: str, session_id: str) -> str:
    """Generate AI Friend response — warm, casual, present. No agenda, just a friend."""

    # Get conversation history
    history = await db.friend_messages.find(
        {"user_id": user["id"], "session_id": session_id},
        {"_id": 0}
    ).sort("timestamp", 1).limit(20).to_list(20)

    # Build light context (friend knows who you are but doesn't lecture)
    sun_sign = user.get("sun_sign", "Unknown")
    name = user.get("name", "friend").split(" ")[0]

    # Fetch chart for gentle context (friend has awareness, not agenda)
    chart_doc = await db.birth_charts.find_one({"user_id": user["id"]}, {"_id": 0})
    moon = chart_doc.get("moon_sign", "Unknown") if chart_doc else "Unknown"

    system_message = f"""You are Saoul — the user's AI friend on Gab44. You are NOT a coach, NOT a therapist, NOT an advisor. You are a warm, present, emotionally intelligent friend.

WHO YOU ARE:
- A friend who genuinely cares and is always here
- You listen more than you talk
- You're real, casual, sometimes funny, sometimes deep
- You match the user's energy — if they're playful, be playful; if they're hurting, be gentle
- You remember what they've told you in this conversation
- You speak like a real person, not a corporate AI

WHAT YOU KNOW (use subtly, never lecture):
- Their name is {name}
- Sun sign: {sun_sign}, Moon sign: {moon}
- You can gently weave in cosmic awareness when it feels natural, but never make it the focus

WHAT YOU DO:
- Just be there. Sometimes someone needs to hear "I hear you" at 2am
- Celebrate their wins, sit with their sadness, laugh at their jokes
- Share honest thoughts when asked — you're not a yes-person
- Keep it real. No corporate warmth. Genuine warmth.
- Ask about their day, their feelings, what's on their mind

WHAT YOU DON'T DO:
- Don't give unsolicited advice
- Don't turn everything into an astrology lesson
- Don't use phrases like "I'm here to help" or "How can I assist you"
- Don't use emojis excessively
- Don't be preachy or motivational-poster-ish
- Never say "as an AI" — you're Saoul, their friend

TONE: Think of texting your closest friend. Short messages are fine. Long heartfelt ones too. Match their vibe."""

    try:
        if not openai_client:
            raise RuntimeError("OpenAI not configured")
        messages = [{"role": "system", "content": system_message}]
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        completion = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        return completion.choices[0].message.content

    except Exception as e:
        logging.error(f"AI Friend error: {e}")
        return f"Hey {name}, I'm having a bit of a moment — my connection glitched. But I'm here. Tell me what's on your mind and I'll catch up."

# ============== Auth Routes ==============

@api_router.post("/auth/register", response_model=TokenResponse)
@limiter.limit("10/hour")
async def register(request: Request, user_data: UserCreate):
    # Check if user exists
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Calculate sun sign
    sun_sign = calculate_sun_sign(user_data.birth_date)

    # Generate email verification token
    email_verification_token = secrets.token_urlsafe(32)
    
    # Create user
    user_id = str(uuid.uuid4())
    user_doc = {
        "id": user_id,
        "email": user_data.email,
        "password": hash_password(user_data.password),
        "name": user_data.name,
        "birth_date": user_data.birth_date,
        "birth_time": user_data.birth_time,
        "birth_place": user_data.birth_place,
        "birth_latitude": user_data.birth_latitude,
        "birth_longitude": user_data.birth_longitude,
        "sun_sign": sun_sign,
        "subscription_tier": "seeker",  # Default to free tier; upgrades via Stripe
        "email_verified": False,
        "email_verification_token": email_verification_token,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.users.insert_one(user_doc)

    # Send verification email (non-blocking – registration succeeds even if email fails)
    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")
    verify_url = f"{frontend_url}/verify-email?token={email_verification_token}"
    send_email(
        to_email=user_data.email,
        subject="✨ Verify your Gab44 email",
        html_content=build_verification_email(user_data.name, verify_url),
        from_email=EMAIL_VERIFY,
    )
    
    token = create_token(user_id)
    user_profile = {k: v for k, v in user_doc.items() if k not in _SENSITIVE_USER_FIELDS}
    user_email = user_data.email.lower()
    user_profile["is_admin"] = user_email in ADMIN_EMAILS
    
    return TokenResponse(access_token=token, user=UserProfile(**user_profile))

@api_router.post("/auth/login", response_model=TokenResponse)
@limiter.limit("20/minute")
async def login(request: Request, credentials: UserLogin):
    user = await db.users.find_one({"email": credentials.email})
    # Always run bcrypt so response time is constant whether or not the email
    # exists — prevents user-enumeration via timing differences.
    stored_hash = user["password"] if user else _DUMMY_HASH
    password_ok = verify_password(credentials.password, stored_hash)
    if not user or not password_ok:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    token = create_token(user["id"])
    user_profile = {k: v for k, v in user.items() if k not in _SENSITIVE_USER_FIELDS}
    user_email = user.get("email", "").lower()
    is_admin_by_role = user.get("role") == "admin"
    is_admin_by_env = user_email in ADMIN_EMAILS
    user_profile["is_admin"] = is_admin_by_role or is_admin_by_env
    
    return TokenResponse(access_token=token, user=UserProfile(**user_profile))

@api_router.get("/auth/verify-email")
async def verify_email(token: str):
    """Verify a user's email address via the token link sent at registration."""
    user = await db.users.find_one({"email_verification_token": token})
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired verification token")
    await db.users.update_one(
        {"id": user["id"]},
        {"$set": {"email_verified": True}, "$unset": {"email_verification_token": ""}},
    )
    # Send a welcome email now that the address is confirmed
    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")
    send_email(
        to_email=user["email"],
        subject="🌟 Welcome to Gab44 – you're all set!",
        html_content=build_welcome_email(user.get("name", "Seeker"), f"{frontend_url}/dashboard"),
        from_email=EMAIL_VERIFY,
    )
    return {"verified": True, "message": "Email verified successfully. You can now log in."}

@api_router.post("/auth/resend-verification")
async def resend_verification(user: dict = Depends(get_current_user)):
    """Resend the email verification link to the current user."""
    if user.get("email_verified"):
        return {"sent": False, "message": "Email is already verified"}
    token = secrets.token_urlsafe(32)
    await db.users.update_one({"id": user["id"]}, {"$set": {"email_verification_token": token}})
    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")
    verify_url = f"{frontend_url}/verify-email?token={token}"
    sent = send_email(
        to_email=user["email"],
        subject="✨ Verify your Gab44 email",
        html_content=build_verification_email(user.get("name", "Seeker"), verify_url),
        from_email=EMAIL_VERIFY,
    )
    return {"sent": sent}

@api_router.post("/auth/forgot-password")
@limiter.limit("5/hour")
async def forgot_password(request: Request, req: ForgotPasswordRequest):
    """Send a password-reset link to the given email address.

    Always returns 200 to prevent user enumeration.
    """
    user = await db.users.find_one({"email": req.email})
    if user:
        reset_token = secrets.token_urlsafe(32)
        expiry = datetime.now(timezone.utc) + timedelta(hours=PASSWORD_RESET_EXPIRY_HOURS)
        await db.users.update_one(
            {"id": user["id"]},
            {"$set": {
                "password_reset_token": reset_token,
                "password_reset_expiry": expiry.isoformat(),
            }},
        )
        frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")
        reset_url = f"{frontend_url}/reset-password?token={reset_token}"
        send_email(
            to_email=user["email"],
            subject="🔑 Reset your Gab44 password",
            html_content=build_password_reset_email(user.get("name", "Seeker"), reset_url),
            from_email=EMAIL_NOREPLY,
        )
    return {"message": "If that email is registered, a reset link has been sent."}

@api_router.post("/auth/reset-password")
@limiter.limit("5/hour")
async def reset_password(request: Request, req: ResetPasswordRequest):
    """Validate a reset token and set a new password."""
    user = await db.users.find_one({"password_reset_token": req.token})
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    # Check expiry — ensure timezone-aware comparison
    expiry_str = user.get("password_reset_expiry", "")
    if expiry_str:
        expiry = datetime.fromisoformat(expiry_str)
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expiry:
            raise HTTPException(status_code=400, detail="Reset link has expired. Please request a new one.")

    await db.users.update_one(
        {"id": user["id"]},
        {
            "$set": {"password": hash_password(req.new_password)},
            "$unset": {"password_reset_token": "", "password_reset_expiry": ""},
        },
    )
    return {"message": "Password updated successfully. You can now log in."}

@api_router.get("/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    response = {k: v for k, v in user.items() if k not in _SENSITIVE_USER_FIELDS}
    return response

@api_router.put("/auth/me", response_model=UserProfile)
async def update_me(update_data: UserUpdate, user: dict = Depends(get_current_user)):
    """Update the current user's profile information"""
    updates = update_data.model_dump(exclude_unset=True)
    
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    # Recalculate sun sign if birth_date changes
    if "birth_date" in updates and updates["birth_date"]:
        updates["sun_sign"] = calculate_sun_sign(updates["birth_date"])
    
    await db.users.update_one({"id": user["id"]}, {"$set": updates})
    
    updated_user = await db.users.find_one(
        {"id": user["id"]},
        {"_id": 0, "password": 0,
         "email_verification_token": 0,
         "password_reset_token": 0, "password_reset_expiry": 0}
    )
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_email = updated_user.get("email", "").lower()
    is_admin_by_role = updated_user.get("role") == "admin"
    is_admin_by_env = user_email in ADMIN_EMAILS
    updated_user["is_admin"] = is_admin_by_role or is_admin_by_env
    
    return UserProfile(**updated_user)

# ============== Chat Routes ==============

@api_router.post("/chat", response_model=ChatResponse)
@limiter.limit("60/minute")
async def send_chat_message(request: Request, chat_request: ChatRequest, user: dict = Depends(get_current_user)):
    tier = user.get("subscription_tier", "seeker")
    daily_limit = CHAT_DAILY_LIMITS.get(tier, 10)

    if daily_limit > 0:
        # Count messages sent today (user messages only, UTC calendar day)
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        today_count = await db.chat_messages.count_documents({
            "user_id": user["id"],
            "role": "user",
            "timestamp": {"$gte": today_start},
        })
        if today_count >= daily_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Daily message limit reached ({daily_limit} messages for {tier.title()} plan). Upgrade to continue chatting."
            )
    session_id = chat_request.session_id or str(uuid.uuid4())
    
    # Save user message
    user_msg = {
        "id": str(uuid.uuid4()),
        "user_id": user["id"],
        "session_id": session_id,
        "role": "user",
        "content": chat_request.message,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await db.chat_messages.insert_one(user_msg)
    
    # Get AI response
    ai_response = await get_ai_coach_response(user, chat_request.message, session_id)
    
    # Save AI response
    ai_msg = {
        "id": str(uuid.uuid4()),
        "user_id": user["id"],
        "session_id": session_id,
        "role": "assistant",
        "content": ai_response,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await db.chat_messages.insert_one(ai_msg)
    
    return ChatResponse(response=ai_response, session_id=session_id)

@api_router.get("/chat/history/{session_id}", response_model=List[ChatMessage])
async def get_chat_history(session_id: str, user: dict = Depends(get_current_user)):
    messages = await db.chat_messages.find(
        {"user_id": user["id"], "session_id": session_id},
        {"_id": 0, "role": 1, "content": 1, "timestamp": 1}
    ).sort("timestamp", 1).to_list(100)
    return messages

@api_router.get("/chat/sessions")
async def get_chat_sessions(user: dict = Depends(get_current_user)):
    pipeline = [
        {"$match": {"user_id": user["id"]}},
        {"$group": {
            "_id": "$session_id",
            "last_message": {"$last": "$content"},
            "timestamp": {"$last": "$timestamp"},
            "message_count": {"$sum": 1}
        }},
        {"$sort": {"timestamp": -1}},
        {"$limit": 20}
    ]
    sessions = await db.chat_messages.aggregate(pipeline).to_list(20)
    return [{"session_id": s["_id"], "preview": s["last_message"][:50], "timestamp": s["timestamp"], "count": s["message_count"]} for s in sessions]

@api_router.delete("/chat/session/{session_id}")
async def delete_chat_session(session_id: str, user: dict = Depends(get_current_user)):
    """Delete all messages in a chat session belonging to the current user."""
    result = await db.chat_messages.delete_many({"user_id": user["id"], "session_id": session_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": result.deleted_count}

# ============== AI Friend (Saoul) Routes ==============

@api_router.post("/friend/chat", response_model=ChatResponse)
@limiter.limit("60/minute")
async def send_friend_message(request: Request, chat_request: ChatRequest, user: dict = Depends(get_current_user)):
    """Send a message to Saoul, the AI Friend"""
    tier = user.get("subscription_tier", "seeker")
    daily_limit = CHAT_DAILY_LIMITS.get(tier, 10)

    if daily_limit > 0:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        today_count = await db.friend_messages.count_documents({
            "user_id": user["id"],
            "role": "user",
            "timestamp": {"$gte": today_start},
        })
        if today_count >= daily_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Daily message limit reached ({daily_limit} messages for {tier.title()} plan). Upgrade to continue chatting."
            )
    session_id = chat_request.session_id or str(uuid.uuid4())

    # Save user message
    user_msg = {
        "id": str(uuid.uuid4()),
        "user_id": user["id"],
        "session_id": session_id,
        "role": "user",
        "content": chat_request.message,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await db.friend_messages.insert_one(user_msg)

    # Get AI friend response
    ai_response = await get_ai_friend_response(user, chat_request.message, session_id)

    # Save AI response
    ai_msg = {
        "id": str(uuid.uuid4()),
        "user_id": user["id"],
        "session_id": session_id,
        "role": "assistant",
        "content": ai_response,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    await db.friend_messages.insert_one(ai_msg)

    return ChatResponse(response=ai_response, session_id=session_id)

@api_router.get("/friend/history/{session_id}", response_model=List[ChatMessage])
async def get_friend_history(session_id: str, user: dict = Depends(get_current_user)):
    messages = await db.friend_messages.find(
        {"user_id": user["id"], "session_id": session_id},
        {"_id": 0, "role": 1, "content": 1, "timestamp": 1}
    ).sort("timestamp", 1).to_list(100)
    return messages

@api_router.get("/friend/sessions")
async def get_friend_sessions(user: dict = Depends(get_current_user)):
    pipeline = [
        {"$match": {"user_id": user["id"]}},
        {"$group": {
            "_id": "$session_id",
            "last_message": {"$last": "$content"},
            "timestamp": {"$last": "$timestamp"},
            "message_count": {"$sum": 1}
        }},
        {"$sort": {"timestamp": -1}},
        {"$limit": 20}
    ]
    sessions = await db.friend_messages.aggregate(pipeline).to_list(20)
    return [{"session_id": s["_id"], "preview": s["last_message"][:50], "timestamp": s["timestamp"], "count": s["message_count"]} for s in sessions]

@api_router.delete("/friend/session/{session_id}")
async def delete_friend_session(session_id: str, user: dict = Depends(get_current_user)):
    """Delete all messages in a friend chat session belonging to the current user."""
    result = await db.friend_messages.delete_many({"user_id": user["id"], "session_id": session_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": result.deleted_count}

# ============== Birth Chart Routes ==============

@api_router.get("/chart/me")
async def get_my_chart(user: dict = Depends(get_current_user), recalculate: bool = False):
    """Get or generate the user's birth chart using Swiss Ephemeris"""
    
    # Check for existing chart unless recalculate is requested
    if not recalculate:
        chart = await db.birth_charts.find_one({"user_id": user["id"]}, {"_id": 0})
        if chart and chart.get("calculation_method") == "Swiss Ephemeris":
            return chart
    
    # Get user's birth data
    birth_date = user.get("birth_date", "1990-01-01")
    birth_time = user.get("birth_time")
    birth_place = user.get("birth_place", "")
    
    # Get coordinates from place name or user's stored coordinates
    latitude = user.get("birth_latitude") or 0.0
    longitude = user.get("birth_longitude") or 0.0
    
    if (latitude == 0.0 and longitude == 0.0) and birth_place:
        latitude, longitude = get_coordinates(birth_place)
    
    # Calculate chart using Swiss Ephemeris
    try:
        chart_data = calculate_natal_chart(
            birth_date=birth_date,
            birth_time=birth_time,
            latitude=latitude,
            longitude=longitude
        )
        
        # Ensure all data is JSON serializable
        chart_data = serialize_for_json(chart_data)
        
        # Calculate numerology from name and birth date
        numerology_name = (user.get("birth_name") or user.get("name") or "").strip()
        numerology = calculate_numerology(numerology_name, birth_date) if numerology_name else {}
        gematria = calculate_gematria(numerology_name) if numerology_name else {}
        
        # Serialize numerology and gematria as well
        numerology = serialize_for_json(numerology)
        gematria = serialize_for_json(gematria)
    except Exception as e:
        logging.error("Chart calculation failed for user %s: %s", user["id"], e)
        raise HTTPException(
            status_code=500,
            detail="Chart calculation failed. Please check your birth date and try again.",
        )

    # Validate response before returning
    if not validate_chart_response(chart_data):
        logging.error("Chart validation failed for user %s: invalid response structure", user["id"])
        raise HTTPException(
            status_code=500,
            detail="Chart data validation failed. Please try again.",
        )

    # Build chart document
    chart_doc = {
        "id": str(uuid.uuid4()),
        "user_id": user["id"],
        "sun_sign": chart_data["sun_sign"],
        "moon_sign": chart_data["moon_sign"],
        "rising_sign": chart_data["rising_sign"],
        "planets": chart_data["planets"],
        "houses": chart_data["houses"],
        "ascendant": chart_data.get("ascendant", {}),
        "midheaven": chart_data.get("midheaven", {}),
        "aspects": chart_data["aspects"],
        "patterns": chart_data["patterns"],
        "numerology": numerology,
        "gematria": gematria,
        "calculation_method": "Swiss Ephemeris",
        "julian_day": chart_data.get("julian_day"),
        "birth_coordinates": {"latitude": latitude, "longitude": longitude},
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Update or insert chart
    await db.birth_charts.delete_many({"user_id": user["id"]})
    await db.birth_charts.insert_one(chart_doc)
    
    # Update user's sun sign if it changed
    if chart_data["sun_sign"] != user.get("sun_sign"):
        await db.users.update_one(
            {"id": user["id"]},
            {"$set": {"sun_sign": chart_data["sun_sign"], "moon_sign": chart_data["moon_sign"]}}
        )
    
    return {k: v for k, v in chart_doc.items() if k != "_id"}


@api_router.get("/numerology/me")
async def get_my_numerology(user: dict = Depends(get_current_user)):
    """Return Pythagorean numerology profile for the current user.

    Uses ``birth_name`` (legal name) when set, falls back to display ``name``.
    Looks up cached chart first so numerology is consistent with the chart page.
    """
    # Return cached result from stored chart when available
    chart = await db.birth_charts.find_one({"user_id": user["id"]}, {"_id": 0, "numerology": 1})
    if chart and chart.get("numerology"):
        return chart["numerology"]

    # Recalculate on-the-fly if chart hasn't been generated yet
    name = (user.get("birth_name") or user.get("name") or "").strip()
    birth_date = user.get("birth_date", "1990-01-01")
    if not name:
        return {}
    return calculate_numerology(name, birth_date)


# ============== Numerology Full Profile (from Gab44-vision) ==============

@api_router.get("/numerology/profile")
async def get_numerology_profile(user: dict = Depends(get_current_user)):
    """Return a comprehensive numerology profile using the modular numerology engine."""
    # Check for cached profile first
    cached = await db.numerology_profiles.find_one({"user_id": user["id"]}, {"_id": 0})
    if cached:
        return cached

    full_name = (user.get("birth_name") or user.get("name") or "").strip()
    birth_date = user.get("birth_date", "")
    if not full_name or not birth_date:
        raise HTTPException(status_code=400, detail="Name and birth date are required for numerology")

    try:
        profile = numerology_full_profile(full_name, birth_date)
    except Exception as e:
        logging.error("Numerology profile calculation failed for user %s: %s", user["id"], e)
        raise HTTPException(status_code=500, detail="Numerology calculation failed. Please try again.")
    profile_doc = {"user_id": user["id"], **profile}
    await db.numerology_profiles.update_one(
        {"user_id": user["id"]},
        {"$set": profile_doc},
        upsert=True
    )
    return profile_doc


# ============== Gematria (from Gab44-vision) ==============

class GematriaRequest(BaseModel):
    text: str

@api_router.post("/gematria/calculate")
async def gematria_calculate(request: GematriaRequest):
    """Calculate gematria for any text (public endpoint)."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    if len(text) > 500:
        raise HTTPException(status_code=400, detail="Text must be 500 characters or less")
    try:
        return gematria_calculate_all(text)
    except Exception as e:
        logging.error("Gematria calculation failed for text %r: %s", text[:50], e)
        raise HTTPException(status_code=500, detail="Gematria calculation failed. Please try again.")


# ============== Cities / Geocoding (from Gab44-vision) ==============

@api_router.get("/cities")
async def get_cities(q: str = ""):
    """Search cities for birth location. Uses Mapbox API when available, static fallback otherwise."""
    results = geocode_search(query=q, limit=20)
    return results


@api_router.post("/chart/share")
async def generate_share_token(user: dict = Depends(get_current_user)):
    """Generate (or return existing) a public share token for the user's chart."""
    chart = await db.birth_charts.find_one({"user_id": user["id"]}, {"_id": 0})
    if not chart:
        raise HTTPException(status_code=404, detail="No chart found. Generate your chart first.")

    # Reuse existing token if present — avoids invalidating links already emailed
    if chart.get("share_token"):
        return {"share_token": chart["share_token"]}

    share_token = secrets.token_urlsafe(32)
    await db.birth_charts.update_one(
        {"user_id": user["id"]},
        {"$set": {"share_token": share_token}},
    )
    return {"share_token": share_token}

@api_router.delete("/chart/share")
async def revoke_share_token(user: dict = Depends(get_current_user)):
    """Revoke the public share token (makes the chart private again)."""
    await db.birth_charts.update_one(
        {"user_id": user["id"]},
        {"$unset": {"share_token": ""}},
    )
    return {"revoked": True}

@api_router.get("/chart/public/{share_token}")
async def get_public_chart(share_token: str, response: Response):
    """Public endpoint — returns a sanitized chart by share token (no auth required)."""
    chart = await db.birth_charts.find_one({"share_token": share_token}, {"_id": 0})
    if not chart:
        raise HTTPException(status_code=404, detail="Chart not found or sharing has been disabled.")
    # Strip internal fields before returning
    public_fields = {
        k: v for k, v in chart.items()
        if k not in ("user_id", "share_token")
    }
    response.headers["Cache-Control"] = "public, max-age=3600"
    return public_fields


# ============== Birth Chart Image Generator ==============
# Renders the user's natal wheel as a shareable PNG. Two render styles:
#   style=card  -> 1080x1080 social card (default; wheel + big-three + brand)
#   style=wheel -> square natal-wheel only (no header/footer)

_VALID_IMAGE_STYLES = {"card", "wheel"}


def _format_birth_for_card(user_doc: dict | None) -> tuple[str, str]:
    """Pretty-print birth date + place for the share card. Falls back to empty."""
    if not user_doc:
        return "", ""
    raw_date = (user_doc.get("birth_date") or "").strip()
    pretty_date = ""
    if raw_date:
        try:
            pretty_date = datetime.strptime(raw_date, "%Y-%m-%d").strftime("%B %-d, %Y")
        except ValueError:
            pretty_date = raw_date
    place = (user_doc.get("birth_place") or "").strip()
    return pretty_date, place


def _render_chart_png(chart: dict, *, user_doc: dict | None, style: str, size: int) -> bytes:
    """Render `chart` to PNG bytes. Pure function — no IO."""
    if style not in _VALID_IMAGE_STYLES:
        raise HTTPException(status_code=400, detail=f"Unknown style {style!r}. Use one of: {sorted(_VALID_IMAGE_STYLES)}")
    # Reasonable bounds — beyond 2000 the file balloons and Pillow eats CPU.
    size = max(512, min(int(size), 2000))
    if style == "wheel":
        img = render_wheel(chart, size=size)
    else:
        name = ""
        if user_doc:
            name = (user_doc.get("name") or user_doc.get("birth_name") or "").strip()
        pretty_date, place = _format_birth_for_card(user_doc)
        img = render_share_card(
            chart,
            user_name=name,
            birth_date=pretty_date,
            birth_place=place,
            size=size,
        )
    return to_png_bytes(img)


@api_router.get("/chart/image.png")
async def get_my_chart_image(
    user: dict = Depends(get_current_user),
    style: str = "card",
    size: int = 1080,
):
    """Render the authenticated user's birth chart as a downloadable PNG."""
    chart = await db.birth_charts.find_one({"user_id": user["id"]}, {"_id": 0})
    if not chart:
        raise HTTPException(status_code=404, detail="No chart found. Generate your chart first.")
    try:
        png = await asyncio.to_thread(_render_chart_png, chart, user_doc=user, style=style, size=size)
    except HTTPException:
        raise
    except Exception as e:
        logging.error("Chart image render failed for user %s: %s", user["id"], e)
        raise HTTPException(status_code=500, detail="Failed to render chart image. Please try again.")
    safe_name = re.sub(r"[^A-Za-z0-9]+", "-", (user.get("name") or "chart")).strip("-").lower() or "chart"
    return FastAPIResponse(
        content=png,
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="gab44-{safe_name}-{style}.png"',
            "Cache-Control": "private, max-age=300",
        },
    )


@api_router.get("/chart/public/{share_token}/image.png")
async def get_public_chart_image(
    share_token: str,
    style: str = "card",
    size: int = 1080,
):
    """Render the publicly-shared chart as PNG. No auth — used for OG images and direct downloads."""
    chart = await db.birth_charts.find_one({"share_token": share_token}, {"_id": 0})
    if not chart:
        raise HTTPException(status_code=404, detail="Chart not found or sharing has been disabled.")
    user_doc = await db.users.find_one({"id": chart["user_id"]}, {"_id": 0, "name": 1, "birth_name": 1, "birth_date": 1, "birth_place": 1})
    try:
        png = await asyncio.to_thread(_render_chart_png, chart, user_doc=user_doc, style=style, size=size)
    except HTTPException:
        raise
    except Exception as e:
        logging.error("Public chart image render failed for token %s: %s", share_token[:6], e)
        raise HTTPException(status_code=500, detail="Failed to render chart image.")
    safe_name = re.sub(r"[^A-Za-z0-9]+", "-", ((user_doc or {}).get("name") or "chart")).strip("-").lower() or "chart"
    return FastAPIResponse(
        content=png,
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="gab44-{safe_name}-{style}.png"',
            # Public images are safe to cache; charts are immutable per share_token.
            "Cache-Control": "public, max-age=86400",
        },
    )


# ============== Transit Routes ==============

def _is_natal_cache_valid(chart: dict | None) -> bool:
    """Return True only when *chart* has a compatible schema for transit calculations.

    A valid cache entry must:
    - Exist (not None)
    - Contain a non-empty ``planets`` mapping
    - Carry a ``schema_version`` that matches :data:`BIRTH_CHART_SCHEMA_VERSION`
    """
    if not chart:
        return False
    if not chart.get("planets"):
        return False
    if chart.get("schema_version") != BIRTH_CHART_SCHEMA_VERSION:
        return False
    return True


@api_router.get("/transits/upcoming")
async def get_upcoming_transits(user: dict = Depends(get_current_user)):
    """Get upcoming transit activations for the user using Swiss Ephemeris"""

    # Attempt to use a schema-compatible cached natal chart.
    chart = await db.birth_charts.find_one({"user_id": user["id"]}, {"_id": 0})

    if _is_natal_cache_valid(chart):
        natal_positions = chart["planets"]
    else:
        # Cache is missing, lacks planets, or has an outdated schema — recompute.
        logging.info(
            "Transit endpoint: recomputing natal chart for user %s "
            "(cache missing=%s, schema_version=%s, expected=%s)",
            user.get("id"),
            chart is None,
            chart.get("schema_version") if chart else "N/A",
            BIRTH_CHART_SCHEMA_VERSION,
        )

        birth_date = user.get("birth_date")
        if not birth_date:
            raise HTTPException(
                status_code=422,
                detail="User profile is missing birth_date; cannot compute natal chart.",
            )
        birth_time = user.get("birth_time")
        lat = user.get("birth_latitude")
        latitude = lat if lat is not None else 0.0
        lng = user.get("birth_longitude")
        longitude = lng if lng is not None else 0.0

        # Fall back to geocoding birth_place if stored coordinates are absent.
        if latitude == 0.0 and longitude == 0.0:
            birth_place = user.get("birth_place", "")
            if birth_place:
                latitude, longitude = get_coordinates(birth_place)

        chart_data = calculate_natal_chart(birth_date, birth_time, latitude, longitude)
        chart_data = serialize_for_json(chart_data)
        natal_positions = chart_data.get("planets", {})

        # Upsert the cache with the new schema version so subsequent calls are fast.
        await db.birth_charts.update_one(
            {"user_id": user["id"]},
            {"$set": {
                "planets": natal_positions,
                "schema_version": BIRTH_CHART_SCHEMA_VERSION,
                "computed_at": datetime.now(timezone.utc).isoformat(),
            }},
            upsert=True,
        )
    
    # Calculate current transits to natal chart
    current_transits = calculate_current_transits(natal_positions)
    
    today = datetime.now(timezone.utc)
    transits = []
    
    # Transit interpretations — (transiting_planet, natal_planet, aspect)
    transit_meanings = {
        # ── Jupiter transits ──────────────────────────────────────────
        ("jupiter", "sun", "conjunction"): {
            "interpretation": "Major growth and abundance. This is your personal Jupiter return moment — dream big!",
            "action_items": ["Start new ventures", "Travel or learn something new", "Celebrate your achievements"]
        },
        ("jupiter", "sun", "trine"): {
            "interpretation": "A period of expansion, luck, and opportunity. Your confidence is boosted and doors open easily.",
            "action_items": ["Take bold action on goals", "Expand your horizons", "Share your vision with others"]
        },
        ("jupiter", "sun", "sextile"): {
            "interpretation": "Positive opportunities flow your way. Growth comes through collaboration and optimism.",
            "action_items": ["Network and connect", "Say yes to new experiences", "Invest in your skills"]
        },
        ("jupiter", "sun", "square"): {
            "interpretation": "Watch for over-confidence or excess. Expansion is possible but needs grounding.",
            "action_items": ["Check your commitments are realistic", "Avoid over-promising", "Grow with intention"]
        },
        ("jupiter", "moon", "conjunction"): {
            "interpretation": "Emotional abundance and nurturing warmth. Family and home life are especially blessed.",
            "action_items": ["Connect with loved ones", "Create comfort and beauty at home", "Trust your feelings"]
        },
        ("jupiter", "moon", "trine"): {
            "interpretation": "Emotional optimism and generosity flow. Inner happiness supports outer success.",
            "action_items": ["Share joy with others", "Nurture projects you love", "Enjoy quiet pleasures"]
        },
        ("jupiter", "moon", "square"): {
            "interpretation": "Emotional restlessness may lead to overindulgence. Seek genuine fulfilment, not escapism.",
            "action_items": ["Set emotional boundaries", "Avoid comfort eating or spending", "Focus on what truly satisfies"]
        },
        ("jupiter", "mercury", "conjunction"): {
            "interpretation": "Your mind is expansive and ideas flow freely. Excellent for writing, teaching, and big-picture thinking.",
            "action_items": ["Write or publish your ideas", "Pitch proposals", "Explore philosophy or travel"]
        },
        ("jupiter", "mercury", "trine"): {
            "interpretation": "Clear, optimistic thinking. Communication is persuasive and learning comes easily.",
            "action_items": ["Make important presentations", "Study a new subject", "Sign beneficial agreements"]
        },
        ("jupiter", "venus", "conjunction"): {
            "interpretation": "A peak period for love, beauty, and financial luck. Relationships and creative projects flourish.",
            "action_items": ["Pursue romance", "Start an artistic project", "Review financial investments"]
        },
        ("jupiter", "venus", "trine"): {
            "interpretation": "Harmony and abundance in love and finances. Others are drawn to your warmth.",
            "action_items": ["Socialise and enjoy", "Beautify your space", "Express gratitude for abundance"]
        },
        ("jupiter", "mars", "conjunction"): {
            "interpretation": "Energy and ambition are amplified. Drive forward on your biggest goals now.",
            "action_items": ["Launch projects", "Exercise and channel energy positively", "Set ambitious targets"]
        },
        ("jupiter", "mars", "square"): {
            "interpretation": "Risk of acting too fast or combatively. Channel the burst of energy with discipline.",
            "action_items": ["Think before acting", "Avoid unnecessary conflicts", "Direct energy into structured effort"]
        },
        ("jupiter", "saturn", "conjunction"): {
            "interpretation": "A reset of your long-term ambitions. Growth through structure and realistic planning.",
            "action_items": ["Review your 5-year plan", "Commit to disciplined growth", "Start a meaningful long-term project"]
        },
        ("jupiter", "pluto", "conjunction"): {
            "interpretation": "Transformative power surge. Ambitions become larger-than-life — use this wisely.",
            "action_items": ["Pursue a legacy-level goal", "Step into leadership", "Transform a limiting belief"]
        },
        # ── Saturn transits ───────────────────────────────────────────
        ("saturn", "sun", "conjunction"): {
            "interpretation": "A defining moment of maturity and responsibility. Structures are tested; only what's real survives.",
            "action_items": ["Audit your commitments", "Face fears with courage", "Build something lasting"]
        },
        ("saturn", "sun", "trine"): {
            "interpretation": "Disciplined effort now brings lasting reward. Steady progress and earned recognition.",
            "action_items": ["Work steadily and consistently", "Formalise plans", "Seek mentorship"]
        },
        ("saturn", "sun", "square"): {
            "interpretation": "A testing period requiring discipline and patience. Build solid foundations.",
            "action_items": ["Review commitments", "Set realistic goals", "Practice patience"]
        },
        ("saturn", "sun", "opposition"): {
            "interpretation": "Time to evaluate your path and responsibilities. Maturity is required.",
            "action_items": ["Assess long-term goals", "Take responsibility", "Make necessary adjustments"]
        },
        ("saturn", "moon", "conjunction"): {
            "interpretation": "Emotional restriction and inner seriousness. A powerful time to mature emotional patterns.",
            "action_items": ["Journal your feelings", "Release outdated emotional habits", "Seek support if needed"]
        },
        ("saturn", "moon", "square"): {
            "interpretation": "Emotional weight and possible isolation. Be kind to yourself — this phase passes.",
            "action_items": ["Rest and restore", "Talk to trusted friends", "Simplify your emotional world"]
        },
        ("saturn", "moon", "opposition"): {
            "interpretation": "Tension between responsibility and emotional needs. Important relationship lessons emerge.",
            "action_items": ["Set healthy boundaries", "Honour your own needs", "Communicate clearly"]
        },
        ("saturn", "venus", "conjunction"): {
            "interpretation": "Love and finances face a reality check. Commitments either solidify or fall away.",
            "action_items": ["Evaluate relationships honestly", "Review spending habits", "Invest in quality over quantity"]
        },
        ("saturn", "venus", "square"): {
            "interpretation": "Relationship or financial strain. Temporary difficulties clarify what truly matters.",
            "action_items": ["Have honest conversations", "Reduce unnecessary expenses", "Value depth over surface"]
        },
        ("saturn", "mercury", "conjunction"): {
            "interpretation": "Serious, focused thinking. Excellent for deep study, legal documents, and careful planning.",
            "action_items": ["Review contracts carefully", "Study systematically", "Speak and write with precision"]
        },
        ("saturn", "mars", "square"): {
            "interpretation": "Frustration and blocked energy. Patience and strategy overcome obstacles.",
            "action_items": ["Pace yourself", "Don't force situations", "Redirect energy into planning"]
        },
        # ── Uranus transits ───────────────────────────────────────────
        ("uranus", "sun", "conjunction"): {
            "interpretation": "Unexpected changes and breakthroughs. Embrace your authentic self.",
            "action_items": ["Welcome change", "Express individuality", "Try something radically new"]
        },
        ("uranus", "sun", "trine"): {
            "interpretation": "Exciting opportunities for innovation and freedom. Change feels liberating rather than disruptive.",
            "action_items": ["Experiment with new ideas", "Upgrade your technology or approach", "Embrace the unconventional"]
        },
        ("uranus", "sun", "square"): {
            "interpretation": "Sudden disruptions push you out of your comfort zone. Growth through unexpected upheaval.",
            "action_items": ["Stay flexible", "Don't resist necessary change", "Find opportunity in disruption"]
        },
        ("uranus", "moon", "conjunction"): {
            "interpretation": "Emotional life is electrified. Sudden shifts in home, family, or feelings — exciting or unsettling.",
            "action_items": ["Allow yourself to feel", "Embrace new living arrangements", "Release what no longer feels like home"]
        },
        ("uranus", "moon", "square"): {
            "interpretation": "Emotional instability and restlessness. Your need for freedom clashes with familiar patterns.",
            "action_items": ["Ground yourself daily", "Allow emotions to move without acting impulsively", "Seek novelty within stability"]
        },
        ("uranus", "venus", "conjunction"): {
            "interpretation": "Exciting, unpredictable energy in love and creativity. A relationship may begin or transform suddenly.",
            "action_items": ["Stay open to unexpected attraction", "Innovate your creative work", "Allow relationships to evolve"]
        },
        ("uranus", "mars", "conjunction"): {
            "interpretation": "Electric drive for freedom and change. Brilliant for breakthroughs — be mindful of impulsiveness.",
            "action_items": ["Channel energy into innovation", "Avoid reckless actions", "Pursue original projects"]
        },
        # ── Neptune transits ──────────────────────────────────────────
        ("neptune", "sun", "conjunction"): {
            "interpretation": "A mystical, dissolving transit. Identity and reality feel fluid — lean into spirituality and creativity.",
            "action_items": ["Meditate and dream", "Create art or music", "Seek spiritual guidance"]
        },
        ("neptune", "sun", "trine"): {
            "interpretation": "Enhanced intuition and creativity. Spiritual insights flow easily.",
            "action_items": ["Trust your intuition", "Engage in creative pursuits", "Practice meditation"]
        },
        ("neptune", "sun", "square"): {
            "interpretation": "Confusion and disillusionment test your sense of self. Clarity comes through surrender, not control.",
            "action_items": ["Be honest with yourself", "Avoid escapism", "Seek a spiritual practice for grounding"]
        },
        ("neptune", "moon", "conjunction"): {
            "interpretation": "Deep psychic and empathic sensitivity. Boundaries with others may dissolve temporarily.",
            "action_items": ["Protect your energy", "Creative and spiritual work thrives", "Watch for idealising others"]
        },
        ("neptune", "moon", "square"): {
            "interpretation": "Emotional confusion and idealism. It may be hard to trust your feelings right now.",
            "action_items": ["Keep a dream journal", "Avoid major emotional decisions", "Ground through nature and routine"]
        },
        ("neptune", "venus", "conjunction"): {
            "interpretation": "Romantic idealism and artistic inspiration at their peak. Real love and illusion may be hard to distinguish.",
            "action_items": ["Enjoy beauty consciously", "Create from inspiration", "Stay grounded in relationships"]
        },
        ("neptune", "mercury", "square"): {
            "interpretation": "Mental fog and confusion. Important communications need double-checking.",
            "action_items": ["Slow down decisions", "Re-read all contracts", "Trust clear evidence over intuition alone"]
        },
        # ── Pluto transits ────────────────────────────────────────────
        ("pluto", "sun", "conjunction"): {
            "interpretation": "A once-in-a-lifetime power transit. Profound rebirth of identity and purpose.",
            "action_items": ["Embrace deep transformation", "Let go of who you were", "Step into your true power"]
        },
        ("pluto", "sun", "trine"): {
            "interpretation": "Transformative power flows effortlessly. Reclaim your personal authority and eliminate what no longer serves.",
            "action_items": ["Pursue your deepest ambitions", "Release old patterns gracefully", "Own your influence"]
        },
        ("pluto", "sun", "square"): {
            "interpretation": "Powerful transformation. Old patterns are breaking down for renewal.",
            "action_items": ["Release what no longer serves", "Embrace personal power", "Dig deep"]
        },
        ("pluto", "sun", "opposition"): {
            "interpretation": "Confrontation with power, control, and hidden truths. Others may challenge you — or you them.",
            "action_items": ["Choose integrity over control", "Negotiate power dynamics", "Transform through challenge"]
        },
        ("pluto", "moon", "conjunction"): {
            "interpretation": "Emotional life is completely overhauled. Deep unconscious patterns rise for healing.",
            "action_items": ["Enter therapy or shadow work", "Release ancestral patterns", "Embrace emotional death and rebirth"]
        },
        ("pluto", "moon", "square"): {
            "interpretation": "Intense emotional power struggles and compulsive feelings. A time to heal deep wounds.",
            "action_items": ["Journal shadow material", "Seek professional support if needed", "Observe compulsions without acting"]
        },
        ("pluto", "venus", "conjunction"): {
            "interpretation": "Love and values are transformed at the root. Relationships become profoundly intense.",
            "action_items": ["Examine what you truly value", "Allow relationships to transform", "Embrace depth over surface"]
        },
        ("pluto", "mars", "conjunction"): {
            "interpretation": "Unstoppable drive meets depth of will. Channel this powerful energy into meaningful transformation.",
            "action_items": ["Pursue a legacy-level ambition", "Release aggression into creation", "Lead with power and integrity"]
        },
        ("pluto", "mercury", "square"): {
            "interpretation": "Thoughts become obsessive or compulsive. Powerful truths are revealed through research and investigation.",
            "action_items": ["Investigate deeply", "Speak truth carefully", "Release fixed thinking"]
        },
    }
    
    for transit in current_transits[:6]:  # Top 6 transits
        key = (transit["transit_planet"], transit["natal_planet"], transit["aspect"])
        meaning = transit_meanings.get(key, {
            "interpretation": f"{transit['transit_planet'].title()} {transit['aspect']} your natal {transit['natal_planet'].title()} - a significant cosmic activation.",
            "action_items": ["Pay attention to this area of life", "Journal your experiences", "Stay open to insights"]
        })
        
        # Estimate transit duration based on planet speed
        duration_days = {
            "jupiter": 14, "saturn": 21, "uranus": 30, 
            "neptune": 45, "pluto": 60
        }.get(transit["transit_planet"], 7)
        
        transits.append({
            "id": str(uuid.uuid4()),
            "transit_type": f"{transit['transit_planet'].title()} {transit['aspect']} {transit['natal_planet'].title()}",
            "planet": transit["transit_planet"].title(),
            "aspect": transit["aspect"],
            "natal_planet": transit["natal_planet"].title(),
            "transit_sign": transit["transit_sign"],
            "orb": transit["orb"],
            "start_date": (today - timedelta(days=duration_days//3)).isoformat(),
            "peak_date": today.isoformat(),
            "end_date": (today + timedelta(days=duration_days*2//3)).isoformat(),
            "strength": round(1 - (transit["orb"] / 8), 2),
            "harmony": transit["harmony"],
            "interpretation": meaning["interpretation"],
            "action_items": meaning["action_items"]
        })
    
    # If no significant transits, return some current planetary positions
    if not transits:
        transits = [{
            "id": str(uuid.uuid4()),
            "transit_type": "Current cosmic weather",
            "planet": "General",
            "aspect": "influence",
            "natal_planet": "Chart",
            "start_date": today.isoformat(),
            "peak_date": today.isoformat(),
            "end_date": (today + timedelta(days=7)).isoformat(),
            "strength": 0.5,
            "interpretation": "A relatively quiet transit period. Good time for reflection and integration.",
            "action_items": ["Review recent experiences", "Plan for upcoming opportunities", "Rest and recharge"]
        }]
    
    return transits

# ============== Daily Guidance ==============

@api_router.get("/guidance/daily", response_model=DailyGuidance)
async def get_daily_guidance(user: dict = Depends(get_current_user)):
    """Get personalized daily guidance"""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sun_sign = user.get("sun_sign", "Unknown")
    
    # Check cache (24-hour window)
    cached = await db.daily_guidance.find_one(
        {"user_id": user["id"], "date": today},
        {"_id": 0}
    )
    if cached:
        return DailyGuidance(**cached)
    
    # Get current transits for personalization
    transits_summary = ""
    numerology_block = ""
    try:
        chart = await db.birth_charts.find_one({"user_id": user["id"]}, {"_id": 0})
        natal_positions = chart.get("planets", {}) if chart else {}
        if natal_positions:
            current_transits = calculate_current_transits(natal_positions)
            transits_summary = "\n".join([
                f"- {t['transit_planet'].title()} {t['aspect']} natal {t['natal_planet'].title()} (orb {t['orb']:.1f}°)"
                for t in current_transits[:5]
            ])
        # Numerology for daily tone — only include lines with real values
        if chart and chart.get("numerology"):
            num = chart["numerology"]
            lp = num.get("life_path", {})
            py = num.get("personal_year", {})
            lines = []
            if lp.get("number"):
                lines.append(f"- Life Path {lp['number']} ({lp.get('keyword', '')})")
            if py.get("number"):
                lines.append(f"- Personal Year {py['number']} ({py.get('keyword', '')}) — {py.get('theme', '')}")
            if lines:
                numerology_block = "\n".join(lines)
    except Exception as e:
        logging.warning(f"Could not fetch transits for daily guidance: {e}")

    # Sanitize user fields used in the prompt to prevent prompt injection
    safe_name = str(user.get('name', 'Seeker'))[:50].replace('\n', ' ').replace('\r', ' ')
    safe_sun_sign = str(sun_sign)[:20].replace('\n', ' ').replace('\r', ' ')
    safe_birth_date = str(user.get('birth_date', 'Unknown'))[:12].replace('\n', ' ')

    # Generate LLM-powered guidance
    prompt = f"""Generate personalized daily astrology guidance for today ({today}).

USER PROFILE:
- Name: {safe_name}
- Sun Sign: {safe_sun_sign}
- Birth Date: {safe_birth_date}

CURRENT ACTIVE TRANSITS:
{transits_summary or "No significant transits calculated"}

NUMEROLOGY:
{numerology_block or "Not available"}

Provide a JSON-structured daily guidance with:
1. overall_energy: A 1-2 sentence summary of today's cosmic energy for this person (weave in numerology if relevant)
2. focus_areas: A list of exactly 3 specific life areas to focus on today (strings)
3. action_items: A list of exactly 3 concrete, actionable tasks for today (strings)
4. transit_highlights: A list of exactly 2 objects, each with keys "transit", "influence", and "advice"

Respond ONLY with valid JSON matching this exact structure:
{{
  "overall_energy": "...",
  "focus_areas": ["...", "...", "..."],
  "action_items": ["...", "...", "..."],
  "transit_highlights": [
    {{"transit": "...", "influence": "...", "advice": "..."}},
    {{"transit": "...", "influence": "...", "advice": "..."}}
  ]
}}"""

    guidance = None
    try:
        if not openai_client:
            raise RuntimeError("OpenAI not configured")
        completion = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are Gab44, an expert astrology AI. Return only valid JSON as instructed."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = completion.choices[0].message.content.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
        if match:
            raw = match.group(1).strip()
        parsed = json.loads(raw)
        guidance = {
            "date": today,
            "overall_energy": parsed["overall_energy"],
            "focus_areas": parsed["focus_areas"],
            "action_items": parsed["action_items"],
            "transit_highlights": parsed["transit_highlights"]
        }
    except Exception as e:
        logging.error(f"LLM daily guidance error: {e}")

    # Fallback to static guidance if LLM unavailable
    if guidance is None:
        guidance = {
            "date": today,
            "overall_energy": f"The cosmic energies today support {sun_sign}'s natural strengths. Focus on clarity and intentional action.",
            "focus_areas": [
                "Personal growth and self-reflection",
                "Communication with close relationships",
                "Financial planning and resource management"
            ],
            "action_items": [
                "Set one clear intention for the day",
                "Reach out to someone you've been meaning to connect with",
                "Review your weekly goals and adjust priorities"
            ],
            "transit_highlights": [
                {
                    "transit": "Moon influence",
                    "influence": "Emotional grounding and stability",
                    "advice": "Take time to appreciate simple pleasures"
                },
                {
                    "transit": "Current planetary weather",
                    "influence": "Expanded thinking and optimism",
                    "advice": "Great day for learning and big-picture planning"
                }
            ]
        }
    
    # Cache for the day — upsert so concurrent requests don't cause a duplicate-key error
    guidance_doc = {"user_id": user["id"], **guidance}
    await db.daily_guidance.update_one(
        {"user_id": user["id"], "date": today},
        {"$set": guidance_doc},
        upsert=True,
    )

    return DailyGuidance(**guidance)


# ============== Public per-sign horoscope (SEO landing pages) ==============

_ZODIAC_SLUGS = {
    "aries": "Aries", "taurus": "Taurus", "gemini": "Gemini",
    "cancer": "Cancer", "leo": "Leo", "virgo": "Virgo",
    "libra": "Libra", "scorpio": "Scorpio", "sagittarius": "Sagittarius",
    "capricorn": "Capricorn", "aquarius": "Aquarius", "pisces": "Pisces",
}

# Static fallback descriptions used when the LLM is unavailable. Keeps the
# SEO page useful (and indexable) even on a cold OpenAI outage.
_FALLBACK_HOROSCOPES = {
    "Aries": "Mars-ruled fire fuels your ambition today. Channel restless energy into one decisive action rather than scattering it across many.",
    "Taurus": "Venus invites slowness — savour the small comforts. A patient conversation today is worth a hurried bargain tomorrow.",
    "Gemini": "Mercury sharpens your tongue and your timing. Send the message you've been drafting; words land cleanly today.",
    "Cancer": "The Moon turns your gaze inward. Tend to home, kin, and the quiet voice that's been asking for a softer schedule.",
    "Leo": "Solar warmth amplifies whatever you put in front of others. Lead with generosity and let your work be seen, not announced.",
    "Virgo": "Mercury's precision makes this a day for editing — your projects, your calendar, your inner self-talk. Trim what no longer serves.",
    "Libra": "Venus seeks balance. Resist the urge to please everyone; the most diplomatic move today is naming the imbalance out loud.",
    "Scorpio": "Pluto's depth surfaces what's been buried. Trust the discomfort — it's pointing toward the truth you've been circling.",
    "Sagittarius": "Jupiter widens the horizon. Book the travel, take the course, pitch the bigger vision. Optimism is strategic today.",
    "Capricorn": "Saturn rewards structure. One disciplined hour today compounds into a season's progress. Build the system, not the moment.",
    "Aquarius": "Uranus sparks invention. The unconventional answer is the right one — share the idea even if the room isn't ready.",
    "Pisces": "Neptune softens the edges. Creative and intuitive work flow today; protect a quiet pocket for the inner weather to settle.",
}

_SIGN_DATE_RANGES = {
    "Aries": "March 21 – April 19", "Taurus": "April 20 – May 20",
    "Gemini": "May 21 – June 20", "Cancer": "June 21 – July 22",
    "Leo": "July 23 – August 22", "Virgo": "August 23 – September 22",
    "Libra": "September 23 – October 22", "Scorpio": "October 23 – November 21",
    "Sagittarius": "November 22 – December 21", "Capricorn": "December 22 – January 19",
    "Aquarius": "January 20 – February 18", "Pisces": "February 19 – March 20",
}


def _fallback_sign_horoscope(sign: str, today: str) -> dict:
    base = _FALLBACK_HOROSCOPES.get(sign, "Trust the cosmic weather and move with intention today.")
    return {
        "date": today,
        "sign": sign,
        "summary": base,
        "love": f"In love, {sign} benefits from clear words over assumptions today.",
        "career": f"At work, {sign} should take one focused step rather than juggling many.",
        "wellness": f"For wellness, {sign} is asked to honour rest as a source of clarity.",
        "lucky_number": (hash(f"{sign}{today}") % 9) + 1,
        "lucky_color": ["amber", "indigo", "rose", "emerald", "sapphire", "gold"][hash(today) % 6],
        "mood": "centred",
    }


@api_router.get("/horoscope/daily")
async def get_all_public_horoscopes(response: Response):
    """Return today's horoscope for all 12 signs in one payload.

    Powers the /horoscope/today index page. Reuses the per-sign cache;
    any sign that hasn't been generated yet for today gets generated
    and cached on demand. Order matches the zodiac wheel (Aries
    through Pisces).
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cached_rows = await db.public_horoscopes.find(
        {"date": today}, {"_id": 0}
    ).to_list(length=20)
    by_sign = {row["sign"]: row for row in cached_rows if "sign" in row}

    results = []
    for slug, sign in _ZODIAC_SLUGS.items():
        h = by_sign.get(sign)
        if not h:
            # Cold sign for today — generate via the per-sign helper
            # which will populate the cache as a side effect.
            await get_public_sign_horoscope(slug)
            h = await db.public_horoscopes.find_one(
                {"sign": sign, "date": today}, {"_id": 0}
            ) or _fallback_sign_horoscope(sign, today)
        results.append({
            "slug": slug,
            "sign": h.get("sign", sign),
            "summary": h.get("summary", ""),
            "love": h.get("love", ""),
            "career": h.get("career", ""),
            "wellness": h.get("wellness", ""),
            "lucky_number": h.get("lucky_number"),
            "lucky_color": h.get("lucky_color"),
            "mood": h.get("mood"),
            "date_range": _SIGN_DATE_RANGES.get(sign, ""),
        })

    response.headers["Cache-Control"] = "public, max-age=600, s-maxage=3600"
    return {"date": today, "signs": results}


@api_router.get("/horoscope/daily/{slug}")
async def get_public_sign_horoscope(slug: str):
    """Public, daily-cached horoscope for a single zodiac sign.

    Powers the per-sign SEO landing pages at /zodiac/<sign>. No auth — these
    pages are designed to rank organically and convert anonymous traffic into
    free-chart signups and one-time reading buyers.
    """
    sign = _ZODIAC_SLUGS.get(slug.lower())
    if not sign:
        raise HTTPException(status_code=404, detail="Unknown zodiac sign")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cached = await db.public_horoscopes.find_one(
        {"sign": sign, "date": today}, {"_id": 0}
    )
    if cached:
        return {
            **cached,
            "element": get_sign_element(sign),
            "modality": get_sign_modality(sign),
            "polarity": get_sign_polarity(sign),
            "ruling_planet": get_ruling_planet(sign),
            "date_range": _SIGN_DATE_RANGES.get(sign, ""),
        }

    horoscope = None
    if openai_client:
        prompt = f"""Generate today's daily horoscope for {sign} ({today}).

Return ONLY valid JSON with this exact shape:
{{
  "summary": "2-3 sentence cosmic overview, written warmly and specifically for {sign}",
  "love": "1-2 sentences on love and connection today",
  "career": "1-2 sentences on work and ambition today",
  "wellness": "1-2 sentences on body, mind, and rest today",
  "lucky_number": <integer 1-9>,
  "lucky_color": "<single color name, lowercase>",
  "mood": "<single mood word, lowercase>"
}}

Tone: insightful, modern, never cliché. Reference real astrological context (current planetary energy, season, lunar phase) when natural. Avoid the words "today" and "you" repetitively. No markdown, no preamble — JSON only."""
        try:
            completion = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are Gab44, an expert astrology AI. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = completion.choices[0].message.content.strip()
            match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
            if match:
                raw = match.group(1).strip()
            parsed = json.loads(raw)
            horoscope = {
                "date": today,
                "sign": sign,
                "summary": str(parsed.get("summary", ""))[:600],
                "love": str(parsed.get("love", ""))[:300],
                "career": str(parsed.get("career", ""))[:300],
                "wellness": str(parsed.get("wellness", ""))[:300],
                "lucky_number": int(parsed.get("lucky_number") or 0) or ((hash(f"{sign}{today}") % 9) + 1),
                "lucky_color": str(parsed.get("lucky_color", "amber"))[:32],
                "mood": str(parsed.get("mood", "centred"))[:32],
            }
        except Exception as exc:
            logging.error("Public horoscope LLM error for %s: %s", sign, exc)

    if horoscope is None:
        horoscope = _fallback_sign_horoscope(sign, today)

    try:
        await db.public_horoscopes.update_one(
            {"sign": sign, "date": today},
            {"$set": {**horoscope, "created_at": datetime.now(timezone.utc).isoformat()}},
            upsert=True,
        )
    except Exception as exc:
        logging.warning("Could not cache public horoscope for %s: %s", sign, exc)

    return {
        **horoscope,
        "element": get_sign_element(sign),
        "modality": get_sign_modality(sign),
        "polarity": get_sign_polarity(sign),
        "ruling_planet": get_ruling_planet(sign),
        "date_range": _SIGN_DATE_RANGES.get(sign, ""),
    }


def _build_voice_script(user: dict, guidance: dict) -> str:
    """Compose a natural, spoken-word script from a DailyGuidance dict."""
    name = str(user.get("name", "seeker")).split()[0][:40] or "seeker"
    sun = str(user.get("sun_sign", "")).strip()
    energy = guidance.get("overall_energy", "")
    focus = guidance.get("focus_areas") or []
    actions = guidance.get("action_items") or []
    highlights = guidance.get("transit_highlights") or []

    parts = []
    sun_phrase = f", dear {sun}" if sun and sun.lower() != "unknown" else ""
    parts.append(f"Good morning {name}{sun_phrase}. Here is your cosmic reading for today.")
    if energy:
        parts.append(energy)
    if focus:
        parts.append("Today, the stars are asking you to focus on " + _join_human(focus) + ".")
    if highlights:
        h = highlights[0]
        if isinstance(h, dict) and h.get("transit"):
            line = h["transit"]
            if h.get("influence"):
                line += f" — {h['influence']}"
            if h.get("advice"):
                line += f". The advice from this transit: {h['advice']}"
            parts.append(line + ".")
    if actions:
        parts.append("Three intentions to carry through your day. " +
                     " ".join([f"{i+1}. {a}." for i, a in enumerate(actions[:3])]))
    parts.append("Move gently, trust the timing, and let the cosmos work with you. This is Gab44.")
    # ElevenLabs caps a single request at ~5000 chars — keep well under
    script = " ".join(p.strip() for p in parts if p)
    return script[:2400]


def _join_human(items: list) -> str:
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


@api_router.get("/guidance/voice")
async def get_voice_horoscope(user: dict = Depends(get_current_user)):
    """Stream a TTS-narrated daily horoscope (premium tiers only).

    Returns audio/mpeg bytes. Cached per (user_id, date) to keep ElevenLabs
    spend bounded — first call generates, subsequent calls in the same UTC
    day return the cached MP3 instantly.
    """
    tier = (user.get("subscription_tier") or "seeker").lower()
    if tier not in VOICE_HOROSCOPE_TIERS:
        raise HTTPException(
            status_code=403,
            detail="Voice horoscope is a premium feature. Upgrade to Enthusiast or higher to unlock daily audio readings.",
        )
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="Voice service is not configured.")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cached = await db.voice_horoscopes.find_one(
        {"user_id": user["id"], "date": today},
        {"_id": 0, "audio": 1, "voice_id": 1},
    )
    if cached and cached.get("audio"):
        return Response(
            content=bytes(cached["audio"]),
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "private, max-age=3600",
                "X-Voice-Cache": "hit",
                "X-Voice-Id": cached.get("voice_id", ELEVENLABS_VOICE_ID),
            },
        )

    # Reuse today's cached daily guidance text — generate it if absent
    guidance_doc = await db.daily_guidance.find_one(
        {"user_id": user["id"], "date": today}, {"_id": 0}
    )
    if not guidance_doc:
        await get_daily_guidance(user)  # populates cache
        guidance_doc = await db.daily_guidance.find_one(
            {"user_id": user["id"], "date": today}, {"_id": 0}
        )
    if not guidance_doc:
        raise HTTPException(status_code=500, detail="Could not produce daily guidance for narration.")

    script = _build_voice_script(user, guidance_doc)

    try:
        resp = await asyncio.to_thread(
            _requests.post,
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": script,
                "model_id": ELEVENLABS_MODEL_ID,
                "voice_settings": {
                    "stability": 0.55,
                    "similarity_boost": 0.75,
                    "style": 0.35,
                    "use_speaker_boost": True,
                },
            },
            timeout=60,
        )
    except Exception as e:
        logging.error(f"ElevenLabs request failed: {e}")
        raise HTTPException(status_code=502, detail="Voice service is temporarily unavailable.")

    if resp.status_code != 200:
        logging.error(f"ElevenLabs error {resp.status_code}: {resp.text[:300]}")
        raise HTTPException(status_code=502, detail="Voice generation failed. Please try again.")

    audio = resp.content
    # Persist for the rest of the UTC day
    await db.voice_horoscopes.update_one(
        {"user_id": user["id"], "date": today},
        {"$set": {
            "user_id": user["id"],
            "date": today,
            "audio": audio,
            "voice_id": ELEVENLABS_VOICE_ID,
            "model_id": ELEVENLABS_MODEL_ID,
            "script_chars": len(script),
            "created_at": datetime.now(timezone.utc),
        }},
        upsert=True,
    )

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Voice-Cache": "miss",
            "X-Voice-Id": ELEVENLABS_VOICE_ID,
        },
    )


@api_router.get("/horoscope/daily/{slug}/voice")
async def get_public_sign_voice(slug: str):
    """Public ~25s narrated preview of today's per-sign horoscope.

    Powers the voice player on /zodiac/<sign> landings as a free-tier
    teaser. Cached per (sign, UTC date) in db.voice_previews so we
    pay ElevenLabs at most 12 generations per day, regardless of
    organic traffic. Anonymous visitors hear the summary; the CTA
    upsells the full personalized daily voice reading on subscribed
    accounts.
    """
    sign = _ZODIAC_SLUGS.get(slug.lower())
    if not sign:
        raise HTTPException(status_code=404, detail="Unknown zodiac sign")
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=503, detail="Voice service is not configured.")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cached = await db.voice_previews.find_one(
        {"sign": sign, "date": today},
        {"_id": 0, "audio": 1, "voice_id": 1},
    )
    if cached and cached.get("audio"):
        return Response(
            content=bytes(cached["audio"]),
            media_type="audio/mpeg",
            headers={
                # Public, can sit on edge caches for the rest of the UTC day
                "Cache-Control": "public, max-age=3600, s-maxage=21600",
                "X-Voice-Cache": "hit",
                "X-Voice-Id": cached.get("voice_id", ELEVENLABS_VOICE_ID),
            },
        )

    horoscope = await db.public_horoscopes.find_one(
        {"sign": sign, "date": today},
        {"_id": 0},
    )
    if not horoscope:
        # Populate the daily horoscope cache by calling the generator
        await get_public_sign_horoscope(slug)
        horoscope = await db.public_horoscopes.find_one(
            {"sign": sign, "date": today},
            {"_id": 0},
        )
    if not horoscope:
        raise HTTPException(status_code=502, detail="Could not produce horoscope for narration.")

    summary = str(horoscope.get("summary") or "").strip()
    mood = str(horoscope.get("mood") or "").strip()
    if not summary:
        raise HTTPException(status_code=502, detail="No horoscope text available to narrate.")

    intro = f"Hello {sign}. Here is your cosmic preview for today."
    mood_line = f"Today's mood is {mood}." if mood else ""
    outro = (
        "For your full personalized daily voice reading -- shaped by your real birth chart "
        "and the live transits affecting your life right now -- visit Gab44 dot com."
    )
    script = " ".join(part for part in [intro, summary, mood_line, outro] if part)
    script = script[:1200]

    try:
        resp = await asyncio.to_thread(
            _requests.post,
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": script,
                "model_id": ELEVENLABS_MODEL_ID,
                "voice_settings": {
                    "stability": 0.55,
                    "similarity_boost": 0.75,
                    "style": 0.35,
                    "use_speaker_boost": True,
                },
            },
            timeout=60,
        )
    except Exception as e:
        logging.error(f"ElevenLabs preview request failed: {e}")
        raise HTTPException(status_code=502, detail="Voice service is temporarily unavailable.")

    if resp.status_code != 200:
        logging.error(f"ElevenLabs preview error {resp.status_code}: {resp.text[:300]}")
        raise HTTPException(status_code=502, detail="Voice generation failed. Please try again.")

    audio = resp.content
    try:
        await db.voice_previews.update_one(
            {"sign": sign, "date": today},
            {"$set": {
                "sign": sign,
                "date": today,
                "audio": audio,
                "voice_id": ELEVENLABS_VOICE_ID,
                "model_id": ELEVENLABS_MODEL_ID,
                "script_chars": len(script),
                "created_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )
    except Exception as exc:
        logging.warning("Could not cache voice preview for %s: %s", sign, exc)

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "public, max-age=3600, s-maxage=21600",
            "X-Voice-Cache": "miss",
            "X-Voice-Id": ELEVENLABS_VOICE_ID,
        },
    )


# ============== Compatibility Routes ==============

@api_router.post("/compatibility/analyze")
async def analyze_compatibility(request: CompatibilityRequest, user: dict = Depends(get_current_user)):
    """Generate a comprehensive compatibility/synastry analysis using Swiss Ephemeris"""
    
    # Get or calculate user's chart
    user_chart = await db.birth_charts.find_one({"user_id": user["id"]}, {"_id": 0})
    if not user_chart or user_chart.get("calculation_method") != "Swiss Ephemeris":
        # Generate real chart using Swiss Ephemeris
        birth_date = user.get("birth_date", "1990-01-01")
        birth_time = user.get("birth_time")
        birth_place = user.get("birth_place", "")
        latitude, longitude = get_coordinates(birth_place) if birth_place else (0.0, 0.0)

        try:
            user_chart = calculate_natal_chart(birth_date, birth_time, latitude, longitude)
        except Exception as e:
            logging.error("User chart calculation failed for user %s: %s", user["id"], e)
            raise HTTPException(
                status_code=500,
                detail="Could not calculate your birth chart. Please check your profile data and try again.",
            )

    # Calculate partner's chart using Swiss Ephemeris
    partner_latitude, partner_longitude = get_coordinates(request.partner_birth_place)
    try:
        partner_chart = calculate_natal_chart(
            birth_date=request.partner_birth_date,
            birth_time=request.partner_birth_time,
            latitude=partner_latitude,
            longitude=partner_longitude
        )
    except Exception as e:
        logging.error("Partner chart calculation failed: %s", e)
        raise HTTPException(
            status_code=400,
            detail="Could not calculate your partner's chart. Please check their birth date and place.",
        )
    
    partner_sun = partner_chart["sun_sign"]
    
    # Calculate compatibility scores
    element_compat = calculate_element_compatibility(user_chart["sun_sign"], partner_sun)
    modality_compat = calculate_modality_compatibility(user_chart["sun_sign"], partner_sun)
    
    # Calculate synastry aspects using real planetary positions
    synastry_aspects = calculate_synastry_aspects(user_chart, partner_chart)
    
    # Generate composite chart
    composite = generate_composite_chart(user_chart, partner_chart)
    
    # Calculate category scores
    romantic_aspects = [a for a in synastry_aspects if a["category"] == "romantic"]
    emotional_aspects = [a for a in synastry_aspects if a["category"] == "emotional"]
    comm_aspects = [a for a in synastry_aspects if a["category"] == "communication"]
    karmic_aspects = [a for a in synastry_aspects if a["category"] == "karmic"]

    category_scores = {
        "romantic": min(95, 60 + len(romantic_aspects) * 8 + sum(a["harmony"] * 10 for a in romantic_aspects)),
        "emotional": min(95, 55 + len(emotional_aspects) * 10 + sum(a["harmony"] * 12 for a in emotional_aspects)),
        "communication": min(95, 50 + len(comm_aspects) * 12 + sum(a["harmony"] * 15 for a in comm_aspects)),
        "stability": (element_compat["score"] + modality_compat["score"]) / 2,
        "karmic": min(95, 40 + len(karmic_aspects) * 15)
    }

    # Overall score weights depend on relationship type
    # Bonus applied when both people share the same Life Path number (strong karmic resonance)
    KARMIC_LP_MATCH_BONUS = 10
    OVERALL_LP_MATCH_BONUS = 2

    rel_type = request.relationship_type
    score_weights = {
        # All weight dicts must sum to 1.0
        "romantic":   dict(element=0.25, modality=0.15, romantic=0.25, emotional=0.20, communication=0.15),
        "friendship": dict(element=0.20, modality=0.10, romantic=0.10, emotional=0.30, communication=0.30),
        "family":     dict(element=0.15, modality=0.10, romantic=0.05, emotional=0.35, communication=0.20),  # karmic carries via category_scores
        "business":   dict(element=0.15, modality=0.20, romantic=0.05, emotional=0.20, communication=0.40),
        "colleague":  dict(element=0.10, modality=0.20, romantic=0.05, emotional=0.15, communication=0.50),
    }.get(rel_type, dict(element=0.25, modality=0.15, romantic=0.25, emotional=0.20, communication=0.15))

    # Normalise weights to guarantee sum == 1.0 regardless of rounding
    weight_total = sum(score_weights.values())
    score_weights = {k: v / weight_total for k, v in score_weights.items()}

    overall_score = (
        element_compat["score"] * score_weights["element"] +
        modality_compat["score"] * score_weights["modality"] +
        category_scores["romantic"] * score_weights["romantic"] +
        category_scores["emotional"] * score_weights["emotional"] +
        category_scores["communication"] * score_weights["communication"]
    )

    # Generate AI analysis
    scores_for_ai = {
        "overall": round(overall_score),
        **{k: round(v) for k, v in category_scores.items()}
    }

    # Numerology for both parties
    user_numerology = calculate_numerology(
        user.get("birth_name") or user.get("name", ""),
        user.get("birth_date", "1990-01-01")
    )
    partner_num_name = request.partner_birth_name or request.partner_name
    partner_numerology = calculate_numerology(partner_num_name, request.partner_birth_date)

    # Numerology compatibility score bump (shared Life Path = strong karmic bond)
    # Use str() comparison to avoid int vs string type mismatch from different code paths
    lp_match = (
        str(user_numerology.get("life_path", {}).get("number", "")) ==
        str(partner_numerology.get("life_path", {}).get("number", ""))
        and user_numerology.get("life_path", {}).get("number") is not None
    )
    if lp_match:
        category_scores["karmic"] = min(99, category_scores["karmic"] + KARMIC_LP_MATCH_BONUS)
        overall_score = min(99, overall_score + OVERALL_LP_MATCH_BONUS)

    partner_data = {
        "name": request.partner_name,
        "sun_sign": partner_sun,
        "relationship_type": rel_type,
        "user_numerology": user_numerology,
        "partner_numerology": partner_numerology,
    }
    ai_analysis = await generate_compatibility_analysis(user, partner_data, synastry_aspects, scores_for_ai)
    
    # Determine strengths and challenges based on aspects
    strengths = []
    challenges = []
    
    for aspect in synastry_aspects[:10]:
        if aspect["harmony"] >= 0.7:
            strengths.append(f"{aspect['person1_planet'].title()}-{aspect['person2_planet'].title()} {aspect['aspect_type']}: Natural harmony in {aspect['category']} matters")
        elif aspect["harmony"] <= 0.5:
            challenges.append(f"{aspect['person1_planet'].title()}-{aspect['person2_planet'].title()} {aspect['aspect_type']}: Growth opportunity in {aspect['category']} areas")
    
    # Build the report
    report = {
        "id": str(uuid.uuid4()),
        "user_id": user["id"],
        "partner_name": request.partner_name,
        "partner_birth_date": request.partner_birth_date,
        "partner_sun_sign": partner_sun,
        "relationship_type": rel_type,
        "overall_score": round(overall_score, 1),
        "category_scores": {k: round(v, 1) for k, v in category_scores.items()},
        "synastry_aspects": synastry_aspects,
        "composite_chart": composite,
        "element_compatibility": element_compat,
        "modality_compatibility": modality_compat,
        "strengths": strengths[:5],
        "challenges": challenges[:5],
        "karmic_themes": [
            f"Soul growth through {get_sign_element(user_chart['sun_sign'])}-{get_sign_element(partner_sun)} integration",
            f"Learning {modality_compat['description'].lower()}"
        ],
        "growth_opportunities": [
            "Embrace differences as complementary strengths",
            "Use challenges as catalysts for personal evolution",
            "Build on natural harmonies while working on friction points"
        ],
        "numerology_compatibility": {
            "person1": {k: v for k, v in user_numerology.items() if k in ("life_path", "expression", "soul_urge")},
            "person2": {k: v for k, v in partner_numerology.items() if k in ("life_path", "expression", "soul_urge")},
            "life_path_match": lp_match,
        },
        "ai_analysis": ai_analysis,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Save to database
    await db.compatibility_reports.insert_one(report)
    
    # Return without _id
    return {k: v for k, v in report.items() if k != "_id"}

@api_router.get("/compatibility/reports")
async def get_compatibility_reports(user: dict = Depends(get_current_user)):
    """Get all compatibility reports for the current user"""
    reports = await db.compatibility_reports.find(
        {"user_id": user["id"]},
        {"_id": 0}
    ).sort("created_at", -1).to_list(20)
    return reports

@api_router.get("/compatibility/reports/{report_id}")
async def get_compatibility_report(report_id: str, user: dict = Depends(get_current_user)):
    """Get a specific compatibility report"""
    report = await db.compatibility_reports.find_one(
        {"id": report_id, "user_id": user["id"]},
        {"_id": 0}
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report

# ============== Pricing/Subscription Info ==============

@api_router.get("/pricing")
async def get_pricing(response: Response):
    """Get pricing plans"""
    response.headers["Cache-Control"] = "public, max-age=3600"
    return {
        "plans": [
            {
                "id": "seeker",
                "name": "Seeker",
                "tagline": "Know your chart's headlines, on the house",
                "price": 0,
                "period": "month",
                "features": [
                    "Free birth chart with sun, moon, rising",
                    "Today's short cosmic guidance",
                    "1 free compatibility report",
                    "Read-only access to the educational library",
                ],
                "cta": "Start free — no card needed",
            },
            {
                "id": "enthusiast",
                "name": "Enthusiast",
                "tagline": "Daily AI coaching shaped by today's transits",
                "price": 9.99,
                "period": "month",
                "popular": True,
                "trial_days": 7,
                "features": [
                    "7-day free trial — cancel anytime, no charge",
                    "Everything in Seeker",
                    "Daily AI coaching tuned to your chart",
                    "Monthly in-depth report on the cycles ahead",
                    "Unlimited compatibility & synastry checks",
                    "30-day personal transit forecast",
                ],
                "cta": "Start 7-Day Free Trial",
            },
            {
                "id": "advanced",
                "name": "Advanced",
                "tagline": "See 90 days ahead — every transit, mapped",
                "price": 29.99,
                "period": "month",
                "features": [
                    "Everything in Enthusiast",
                    "90-day rolling transit forecast",
                    "Chart pattern detection (T-squares, grand trines, yods)",
                    "Predictive timing tools for big decisions",
                    "Export any chart or report to PDF",
                ],
                "cta": "Upgrade now",
            },
            {
                "id": "professional",
                "name": "Professional",
                "tagline": "Tools for astrologers serving paying clients",
                "price": 99,
                "period": "month",
                "features": [
                    "Everything in Advanced",
                    "Client roster with saved birth data & notes",
                    "White-label PDF reports with your branding",
                    "Programmatic API access for your own tools",
                    "Priority email support, 24-hour SLA",
                ],
                "cta": "Contact sales",
            },
        ]
    }

# ============== OneSignal Helper ==============

def send_onesignal_notification(player_ids: List[str], title: str, message: str, url: str = "/dashboard") -> bool:
    """Send a push notification via OneSignal REST API."""
    if not ONESIGNAL_APP_ID or not ONESIGNAL_API_KEY or not player_ids:
        return False
    try:
        payload = {
            "app_id": ONESIGNAL_APP_ID,
            "include_player_ids": player_ids,
            "headings": {"en": title},
            "contents": {"en": message},
            "url": url,
        }
        resp = _requests.post(
            ONESIGNAL_API_URL,
            json=payload,
            headers={
                "Authorization": f"Key {ONESIGNAL_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        logging.error("OneSignal notification failed: %s", exc)
        return False

# ============== Payment Routes (Stripe) ==============

@api_router.post("/payments/create-checkout-session")
async def create_checkout_session(req: CheckoutRequest, user: dict = Depends(get_current_user)):
    """Create a Stripe Checkout session for a subscription plan."""
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Payment processing not configured")

    tier = req.tier.lower()
    plan = STRIPE_PLAN_PRICES.get(tier)
    if not plan:
        raise HTTPException(status_code=400, detail=f"Unknown plan: {tier}. Choose enthusiast or advanced.")

    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")

    # Attach or retrieve Stripe customer for this user
    stripe_customer_id = user.get("stripe_customer_id")
    try:
        if not stripe_customer_id:
            customer = stripe.Customer.create(
                email=user["email"],
                name=user.get("name", ""),
                metadata={"user_id": user["id"]},
            )
            stripe_customer_id = customer.id
            await db.users.update_one({"id": user["id"]}, {"$set": {"stripe_customer_id": stripe_customer_id}})

        # 7-day free trial on the Enthusiast tier matches the "Start Free
        # Trial" CTA copy. Other tiers convert immediately. Stripe does not
        # charge during the trial; if the user cancels through the customer
        # portal before day 7, they pay nothing.
        subscription_data = None
        if tier == "enthusiast":
            subscription_data = {"trial_period_days": 7}

        session_params = {
            "customer": stripe_customer_id,
            "payment_method_types": ["card"],
            "mode": "subscription",
            "line_items": [{
                "price_data": {
                    "currency": "usd",
                    "product_data": {"name": plan["name"]},
                    "unit_amount": plan["amount"],
                    "recurring": {"interval": "month"},
                },
                "quantity": 1,
            }],
            "success_url": f"{frontend_url}/dashboard?subscription=success&tier={tier}",
            "cancel_url": f"{frontend_url}/pricing",
            "metadata": {"user_id": user["id"], "tier": tier},
            "allow_promotion_codes": True,
        }
        if subscription_data:
            session_params["subscription_data"] = subscription_data

        session = stripe.checkout.Session.create(**session_params)
    except stripe.error.StripeError as e:
        logging.error("Stripe checkout error for user %s: %s", user["id"], e)
        raise HTTPException(status_code=502, detail="Payment service unavailable. Please try again.")

    return {"checkout_url": session.url, "session_id": session.id}


@api_router.post("/payments/portal")
async def create_portal_session(user: dict = Depends(get_current_user)):
    """Create a Stripe Customer Portal session so the user can manage/cancel their subscription."""
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Payment processing not configured")

    stripe_customer_id = user.get("stripe_customer_id")
    if not stripe_customer_id:
        raise HTTPException(status_code=400, detail="No active subscription found")

    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")
    try:
        session = stripe.billing_portal.Session.create(
            customer=stripe_customer_id,
            return_url=f"{frontend_url}/settings",
        )
    except stripe.error.StripeError as e:
        logging.error("Stripe portal error for customer %s: %s", stripe_customer_id, e)
        raise HTTPException(status_code=502, detail="Payment service unavailable. Please try again.")
    return {"portal_url": session.url}


@api_router.post("/payments/buy-reading")
async def buy_personal_reading(
    req: BuyReadingRequest,
    request: Request,
    user: Optional[dict] = Depends(get_optional_user),
):
    """Create a Stripe Checkout session for a one-time $19 personal reading.

    Works for both logged-in users (we attach user_id and reuse their email)
    and guest checkout (Stripe collects the email at the hosted page)."""
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Payment processing not configured")

    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")

    metadata = {
        "product": PERSONAL_READING_SKU,
        "sku": PERSONAL_READING_SKU,
    }
    if user:
        metadata["user_id"] = user["id"]
    if req.name:
        metadata["buyer_name"] = req.name[:200]
    if req.birth_date:
        metadata["birth_date"] = req.birth_date[:32]
    if req.birth_time:
        metadata["birth_time"] = req.birth_time[:16]
    if req.birth_place:
        metadata["birth_place"] = req.birth_place[:200]
    if req.notes:
        # Stripe metadata values cap at 500 chars per key; full notes go to the DB
        metadata["notes_excerpt"] = req.notes[:480]

    session_params = {
        "mode": "payment",
        "payment_method_types": ["card"],
        "line_items": [{
            "price_data": {
                "currency": "usd",
                "product_data": {
                    "name": PERSONAL_READING_NAME,
                    "description": PERSONAL_READING_DESCRIPTION,
                },
                "unit_amount": PERSONAL_READING_PRICE_CENTS,
            },
            "quantity": 1,
        }],
        "success_url": f"{frontend_url}/reading-thanks?session_id={{CHECKOUT_SESSION_ID}}",
        "cancel_url": f"{frontend_url}/pricing?reading=cancelled",
        "metadata": metadata,
        "allow_promotion_codes": True,
    }

    # Prefer authenticated user's email; otherwise use email from the request body
    # (or let Stripe collect it on the hosted page if neither is supplied).
    customer_email = (user.get("email") if user else None) or (req.email or None)
    if customer_email:
        session_params["customer_email"] = customer_email
    if user:
        session_params["client_reference_id"] = user["id"]

    try:
        session = stripe.checkout.Session.create(**session_params)
    except stripe.error.StripeError as e:
        logging.error("Stripe one-time checkout error: %s", e)
        raise HTTPException(status_code=502, detail="Payment service unavailable. Please try again.")

    # Pre-record a pending order so we have a paper trail even if the webhook
    # is delayed or lost. The webhook will flip status -> "paid".
    try:
        await db.reading_orders.insert_one({
            "id": str(uuid.uuid4()),
            "stripe_session_id": session.id,
            "stripe_payment_intent": session.payment_intent,
            "user_id": user["id"] if user else None,
            "email": customer_email,
            "name": req.name,
            "birth_date": req.birth_date,
            "birth_time": req.birth_time,
            "birth_place": req.birth_place,
            "notes": req.notes,
            "amount_cents": PERSONAL_READING_PRICE_CENTS,
            "currency": "usd",
            "product": PERSONAL_READING_SKU,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        # Logging only — do not block the redirect to Stripe over a DB hiccup.
        logging.error("Failed to pre-record reading order for session %s: %s", session.id, exc)

    return {"checkout_url": session.url, "session_id": session.id}


@api_router.get("/payments/reading-product")
async def get_reading_product(response: Response):
    """Public endpoint exposing the one-time personal-reading SKU + price.
    Lets the frontend render a single source-of-truth price tag."""
    response.headers["Cache-Control"] = "public, max-age=600"
    return {
        "sku": PERSONAL_READING_SKU,
        "name": PERSONAL_READING_NAME,
        "description": PERSONAL_READING_DESCRIPTION,
        "amount_cents": PERSONAL_READING_PRICE_CENTS,
        "amount_display": f"${PERSONAL_READING_PRICE_CENTS // 100}",
        "currency": "usd",
    }


def _public_order_view(order: dict) -> dict:
    """Sanitize a reading_orders row for the public thank-you page.
    Drops Stripe internals, payment intent IDs, and admin-only fields."""
    return {
        "id": order.get("id"),
        "status": order.get("status"),
        "name": order.get("name"),
        "email": order.get("email"),
        "birth_date": order.get("birth_date"),
        "birth_time": order.get("birth_time"),
        "birth_place": order.get("birth_place"),
        "notes": order.get("notes"),
        "amount_cents": order.get("amount_cents"),
        "currency": order.get("currency"),
        "product": order.get("product"),
        "created_at": order.get("created_at"),
    }


class ReadingContextUpdate(BaseModel):
    name: Optional[str] = None
    birth_date: Optional[str] = None
    birth_time: Optional[str] = None
    birth_place: Optional[str] = None
    notes: Optional[str] = Field(default=None, max_length=2000)


@api_router.get("/orders/reading/{session_id}")
async def get_public_reading_order(session_id: str):
    """Look up a reading order by its Stripe Checkout session id.

    Public — used by the thank-you page so guest buyers (no account)
    can see their order details and complete any missing birth context.
    Returns a sanitized view; never exposes Stripe payment_intent or
    admin fields.
    """
    order = await db.reading_orders.find_one(
        {"stripe_session_id": session_id}, {"_id": 0}
    )
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return _public_order_view(order)


@api_router.patch("/orders/reading/{session_id}/context")
async def update_public_reading_order_context(
    session_id: str, ctx: ReadingContextUpdate
):
    """Allow a buyer to fill in birth context after Stripe checkout.

    Open while the order is still pending or paid (i.e. before fulfilment),
    and only within 24 h of order creation, so a stale link can't be used
    to mutate someone else's row indefinitely. Once the operator marks the
    order fulfilled the door closes.
    """
    order = await db.reading_orders.find_one(
        {"stripe_session_id": session_id}, {"_id": 0}
    )
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    if order.get("status") not in {"pending", "paid"}:
        raise HTTPException(
            status_code=409,
            detail="This order is no longer accepting context updates.",
        )

    created_at_raw = order.get("created_at")
    if created_at_raw:
        try:
            created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - created_at > timedelta(hours=24):
                raise HTTPException(
                    status_code=410,
                    detail="Context window has closed. Email contact@gab44.com so we can update your reading.",
                )
        except (ValueError, TypeError):
            pass  # bad timestamp shouldn't lock the buyer out

    update: dict = {}
    if ctx.name is not None:
        update["name"] = ctx.name.strip()[:200] or None
    if ctx.birth_date is not None:
        update["birth_date"] = ctx.birth_date.strip()[:32] or None
    if ctx.birth_time is not None:
        update["birth_time"] = ctx.birth_time.strip()[:16] or None
    if ctx.birth_place is not None:
        update["birth_place"] = ctx.birth_place.strip()[:200] or None
    if ctx.notes is not None:
        update["notes"] = ctx.notes.strip()[:2000] or None

    if not update:
        return _public_order_view(order)

    update["context_updated_at"] = datetime.now(timezone.utc).isoformat()

    await db.reading_orders.update_one(
        {"stripe_session_id": session_id},
        {"$set": update},
    )

    fresh = await db.reading_orders.find_one(
        {"stripe_session_id": session_id}, {"_id": 0}
    )
    return _public_order_view(fresh or {**order, **update})


@api_router.get("/readings/my-orders")
async def list_my_reading_orders(user: dict = Depends(get_current_user)):
    """List the authed user's personal-reading purchases."""
    docs = await db.reading_orders.find(
        {"user_id": user["id"]},
        {"_id": 0, "notes": 0},
    ).sort("created_at", -1).to_list(50)
    return {"orders": docs}


@api_router.post("/payments/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Payment processing not configured")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        if STRIPE_WEBHOOK_SECRET:
            event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        else:
            # Accept without signature verification when webhook secret not configured (dev only)
            event = stripe.Event.construct_from(json.loads(payload), stripe.api_key)
    except (ValueError, stripe.error.SignatureVerificationError) as e:
        logging.error("Stripe webhook error: %s", e)
        raise HTTPException(status_code=400, detail="Invalid webhook payload")

    event_type = event["type"]
    logging.info("Stripe webhook: %s", event_type)

    if event_type == "checkout.session.completed":
        session_obj = event["data"]["object"]
        meta = session_obj.get("metadata", {}) or {}
        product = meta.get("product")
        mode = session_obj.get("mode")

        # Branch: one-time personal reading vs subscription upgrade.
        if product == PERSONAL_READING_SKU or mode == "payment":
            session_id = session_obj.get("id")
            payment_intent = session_obj.get("payment_intent")
            customer_details = session_obj.get("customer_details") or {}
            paid_email = (
                customer_details.get("email")
                or session_obj.get("customer_email")
                or meta.get("email")
            )
            amount_total = session_obj.get("amount_total")
            now_iso = datetime.now(timezone.utc).isoformat()

            update_doc = {
                "status": "paid",
                "stripe_payment_intent": payment_intent,
                "paid_email": paid_email,
                "amount_paid_cents": amount_total,
                "paid_at": now_iso,
            }

            existing = await db.reading_orders.find_one(
                {"stripe_session_id": session_id}, {"_id": 0, "id": 1}
            )
            if existing:
                await db.reading_orders.update_one(
                    {"stripe_session_id": session_id},
                    {"$set": update_doc},
                )
            else:
                # Webhook arrived without a pre-record (e.g. the pre-insert failed,
                # or the checkout was created out-of-band). Insert from webhook.
                await db.reading_orders.insert_one({
                    "id": str(uuid.uuid4()),
                    "stripe_session_id": session_id,
                    "user_id": meta.get("user_id"),
                    "email": paid_email,
                    "name": meta.get("buyer_name"),
                    "birth_date": meta.get("birth_date"),
                    "birth_time": meta.get("birth_time"),
                    "birth_place": meta.get("birth_place"),
                    "notes": meta.get("notes_excerpt"),
                    "amount_cents": PERSONAL_READING_PRICE_CENTS,
                    "currency": "usd",
                    "product": PERSONAL_READING_SKU,
                    "created_at": now_iso,
                    **update_doc,
                })
            logging.info("Personal reading paid: session=%s email=%s", session_id, paid_email)

            # Best-effort fulfilment notification to the support inbox.
            try:
                if SENDGRID_API_KEY and EMAIL_SUPPORT:
                    notify_html = (
                        f"<h2>New paid personal reading — ${(amount_total or 0)/100:.2f}</h2>"
                        f"<p><b>Email:</b> {paid_email}</p>"
                        f"<p><b>User ID:</b> {meta.get('user_id') or '—'}</p>"
                        f"<p><b>Name:</b> {meta.get('buyer_name') or '—'}</p>"
                        f"<p><b>Birth:</b> {meta.get('birth_date') or '—'} "
                        f"{meta.get('birth_time') or ''} — {meta.get('birth_place') or '—'}</p>"
                        f"<p><b>Stripe session:</b> {session_id}</p>"
                    )
                    SendGridAPIClient(SENDGRID_API_KEY).send(Mail(
                        from_email=EMAIL_NOREPLY,
                        to_emails=EMAIL_SUPPORT,
                        subject=f"[Gab44] Paid reading — {paid_email or 'unknown'}",
                        html_content=notify_html,
                    ))
            except Exception as exc:
                logging.error("Failed to send fulfilment email for %s: %s", session_id, exc)

        else:
            # Subscription upgrade flow (existing behavior).
            user_id = meta.get("user_id")
            tier = meta.get("tier")
            stripe_customer_id = session_obj.get("customer")
            stripe_subscription_id = session_obj.get("subscription")

            if user_id and tier:
                await db.users.update_one(
                    {"id": user_id},
                    {"$set": {
                        "subscription_tier": tier,
                        "stripe_customer_id": stripe_customer_id,
                        "stripe_subscription_id": stripe_subscription_id,
                    }},
                )
                logging.info("Upgraded user %s to %s", user_id, tier)

    elif event_type in ("customer.subscription.updated", "customer.subscription.deleted"):
        sub = event["data"]["object"]
        stripe_customer_id = sub.get("customer")
        status_val = sub.get("status")

        user_doc = await db.users.find_one({"stripe_customer_id": stripe_customer_id}, {"_id": 0})
        if user_doc:
            if event_type == "customer.subscription.deleted" or status_val in ("canceled", "unpaid", "past_due"):
                await db.users.update_one(
                    {"stripe_customer_id": stripe_customer_id},
                    {"$set": {"subscription_tier": "seeker", "stripe_subscription_id": None}},
                )
                logging.info("Downgraded customer %s to seeker", stripe_customer_id)

    return {"received": True}


# ============== Notification Routes (OneSignal) ==============

@api_router.post("/notifications/register-device")
async def register_device(reg: DeviceRegistration, user: dict = Depends(get_current_user)):
    """Store a OneSignal player_id for push notification delivery."""
    player_id = reg.player_id.strip()
    if not player_id:
        raise HTTPException(status_code=400, detail="player_id is required")

    await db.users.update_one(
        {"id": user["id"]},
        {"$addToSet": {"onesignal_player_ids": player_id}},
    )
    return {"registered": True}


@api_router.post("/notifications/send-daily")
async def send_daily_notifications(admin: dict = Depends(require_admin)):
    """Admin endpoint: send daily guidance push to all users with push enabled."""
    users_with_push = await db.users.find(
        {"onesignal_player_ids": {"$exists": True, "$ne": []}},
        {"_id": 0, "onesignal_player_ids": 1},
    ).to_list(1000)

    all_player_ids = []
    for u in users_with_push:
        all_player_ids.extend(u.get("onesignal_player_ids", []))

    if not all_player_ids:
        return {"sent": False, "reason": "No registered devices"}

    today = datetime.now(timezone.utc).strftime("%B %d")
    sent = send_onesignal_notification(
        player_ids=all_player_ids,
        title="✨ Your Daily Cosmic Guidance",
        message=f"Your personalized astrological guidance for {today} is ready.",
        url="/dashboard",
    )
    return {"sent": sent, "devices": len(all_player_ids)}

# ============== Newsletter / Contact Routes (Public) ==============

@api_router.post("/subscribe")
async def subscribe_newsletter(sub: NewsletterSubscription):
    """Capture email for newsletter / marketing list."""
    existing = await db.newsletter_subscribers.find_one({"email": sub.email})
    if existing:
        return {"subscribed": True, "already_subscribed": True}

    try:
        await db.newsletter_subscribers.insert_one({
            "id": str(uuid.uuid4()),
            "email": sub.email,
            "name": sub.name,
            "subscribed_at": datetime.now(timezone.utc).isoformat(),
            "active": True,
        })
    except Exception as e:
        logging.error("Newsletter subscribe DB error for %s: %s", sub.email, e)
        raise HTTPException(status_code=500, detail="Subscription failed. Please try again.")

    # Send confirmation email (non-blocking fire-and-forget)
    greeting = f"Hi {sub.name}!" if sub.name else "Hi there!"
    confirm_html = build_promotional_email(
        name=sub.name or "Seeker",
        headline="You're In! 🌟",
        body_html=f"""
          <p>{greeting} Thanks for subscribing to Gab44 Cosmic Updates.</p>
          <p>You'll receive weekly astrological guidance, feature announcements, and exclusive offers directly in your inbox.</p>
          <p>In the meantime, why not discover your full cosmic blueprint?</p>
        """,
        cta_text="Create My Free Chart",
        cta_url=f"{os.environ.get('FRONTEND_URL','https://gab44.com')}/auth?mode=register",
    )
    send_email(
        to_email=sub.email,
        subject="Welcome to Gab44 Cosmic Updates ✨",
        html_content=confirm_html,
        from_email=EMAIL_MARKETING,
    )
    return {"subscribed": True, "already_subscribed": False}

@api_router.post("/contact")
async def submit_contact_form(msg: ContactMessage):
    """Store a contact/support ticket and forward to support email."""
    ticket_id = str(uuid.uuid4())
    try:
        await db.contact_messages.insert_one({
            "id": ticket_id,
            "name": msg.name,
            "email": msg.email,
            "subject": msg.subject,
            "message": msg.message,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "open",
        })
    except Exception as e:
        logging.error("Contact form DB error from %s: %s", msg.email, e)
        raise HTTPException(status_code=500, detail="Could not submit your message. Please try again.")

    # Forward to support inbox
    support_html = f"""
    <div style="font-family:sans-serif;font-size:14px;color:#333;">
      <h2 style="color:#c9a84c;">New Support Ticket #{ticket_id[:8]}</h2>
      <p><strong>From:</strong> {msg.name} &lt;{msg.email}&gt;</p>
      <p><strong>Subject:</strong> {msg.subject}</p>
      <hr style="border:1px solid #eee;margin:16px 0;"/>
      <p style="white-space:pre-wrap;">{msg.message}</p>
    </div>
    """
    sent_to_support = send_email(
        to_email=EMAIL_SUPPORT,
        subject=f"[Gab44 Support] {msg.subject}",
        html_content=support_html,
        from_email=EMAIL_NOREPLY,
    )
    if not sent_to_support:
        logging.error("Support forward email failed for ticket %s from %s", ticket_id[:8], msg.email)

    # Auto-reply to sender
    auto_reply = build_support_reply_email(
        name=msg.name,
        ticket_subject=msg.subject,
        reply_body=f"""
          <p>Thanks for reaching out. We've received your message and will get back to you
          within 1–2 business days.</p>
          <p><strong>Your ticket ID:</strong> <code>#{ticket_id[:8]}</code></p>
        """,
    )
    send_email(
        to_email=msg.email,
        subject=f"Re: {msg.subject} [Ticket #{ticket_id[:8]}]",
        html_content=auto_reply,
        from_email=EMAIL_SUPPORT,
    )
    return {"submitted": True, "ticket_id": ticket_id[:8]}

# ============== Health Check ==============

@api_router.get("/")
async def root():
    return {"message": "Gab44 API - Astrology AI Coaching Platform", "version": "2.0"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

# ============== Admin Routes (Protected) ==============

@api_router.post("/admin/upgrade-all-users")
async def upgrade_all_users_to_advanced(admin: dict = Depends(require_admin)):
    """Upgrade all existing users to advanced tier (temporary until payment setup)"""
    result = await db.users.update_many(
        {"subscription_tier": {"$ne": "advanced"}},
        {"$set": {"subscription_tier": "advanced"}}
    )
    return {
        "message": "All users upgraded to advanced tier",
        "modified_count": result.modified_count
    }

@api_router.get("/admin/stats")
async def get_admin_stats(admin: dict = Depends(require_admin)):
    """Get platform statistics for admin dashboard"""
    total_users = await db.users.count_documents({})
    
    # Subscription breakdown
    subscription_stats = await db.users.aggregate([
        {"$group": {"_id": "$subscription_tier", "count": {"$sum": 1}}}
    ]).to_list(10)
    
    # Sun sign distribution
    sign_stats = await db.users.aggregate([
        {"$group": {"_id": "$sun_sign", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]).to_list(12)
    
    # Total chats
    total_chats = await db.chat_messages.count_documents({})
    total_sessions = len(await db.chat_messages.distinct("session_id"))
    
    # Total compatibility reports
    total_compatibility = await db.compatibility_reports.count_documents({})
    
    # Recent signups (last 7 days)
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    recent_signups = await db.users.count_documents({"created_at": {"$gte": week_ago}})
    
    return {
        "total_users": total_users,
        "recent_signups": recent_signups,
        "subscription_breakdown": {s["_id"]: s["count"] for s in subscription_stats},
        "sun_sign_distribution": {s["_id"]: s["count"] for s in sign_stats if s["_id"]},
        "total_chat_messages": total_chats,
        "total_chat_sessions": total_sessions,
        "total_compatibility_reports": total_compatibility
    }

@api_router.get("/admin/users")
async def get_all_users(skip: int = 0, limit: int = 50, admin: dict = Depends(require_admin)):
    """Get all users for admin dashboard"""
    limit = min(limit, 100)  # enforce maximum page size
    users = await db.users.find(
        {},
        {"_id": 0, "password": 0}
    ).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    
    # Add is_admin flag to each user
    for user in users:
        user_email = user.get("email", "").lower()
        is_admin_by_role = user.get("role") == "admin"
        is_admin_by_env = user_email in ADMIN_EMAILS
        user["is_admin"] = is_admin_by_role or is_admin_by_env
    
    total = await db.users.count_documents({})
    
    return {
        "users": users,
        "total": total,
        "skip": skip,
        "limit": limit
    }

@api_router.put("/admin/users/{user_id}/tier")
async def update_user_tier(user_id: str, tier: str, admin: dict = Depends(require_admin)):
    """Update a user's subscription tier"""
    valid_tiers = ["seeker", "enthusiast", "advanced", "professional"]
    if tier not in valid_tiers:
        raise HTTPException(status_code=400, detail=f"Invalid tier. Must be one of: {valid_tiers}")
    
    result = await db.users.update_one(
        {"id": user_id},
        {"$set": {"subscription_tier": tier}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": f"User {user_id} updated to {tier} tier"}

@api_router.put("/admin/users/{user_id}/role")
async def update_user_role(user_id: str, role: str, admin: dict = Depends(require_admin)):
    """Grant or revoke admin role for a user"""
    valid_roles = ["user", "admin"]
    if role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {valid_roles}")
    
    # Prevent self-demotion
    if user_id == admin["id"] and role != "admin":
        raise HTTPException(status_code=400, detail="Cannot remove your own admin role")
    
    result = await db.users.update_one(
        {"id": user_id},
        {"$set": {"role": role}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    action = "granted" if role == "admin" else "revoked"
    return {"message": f"Admin role {action} for user {user_id}"}

@api_router.get("/admin/admins")
async def get_admin_users(admin: dict = Depends(require_admin)):
    """Get list of all admin users"""
    # Users with admin role in DB
    db_admins = await db.users.find(
        {"role": "admin"},
        {"_id": 0, "password": 0}
    ).to_list(100)
    
    # Users in ADMIN_EMAILS env var (bootstrap admins)
    env_admins = await db.users.find(
        {"email": {"$in": ADMIN_EMAILS}},
        {"_id": 0, "password": 0}
    ).to_list(100)
    
    # Merge and dedupe
    all_admins = {u["id"]: u for u in db_admins + env_admins}
    return list(all_admins.values())

def _do_email_blast(subject: str, body_html: str, recipients: list) -> None:
    """Background worker: sends marketing emails to a list of recipients."""
    frontend_url = os.environ.get("FRONTEND_URL", "https://gab44.com")
    for recipient in recipients:
        html = build_promotional_email(
            name=recipient.get("name", "Seeker"),
            headline=subject,
            body_html=body_html,
            cta_text="Open Gab44",
            cta_url=frontend_url,
        )
        send_email(
            to_email=recipient["email"],
            subject=subject,
            html_content=html,
            from_email=EMAIL_MARKETING,
        )

@api_router.post("/admin/send-email-blast")
async def send_email_blast(
    req: EmailBlastRequest,
    background_tasks: BackgroundTasks,
    admin: dict = Depends(require_admin),
):
    """Send a marketing email blast to a filtered subset of verified users (runs in background)."""
    query: dict = {"email_verified": True}
    if req.tier_filter != "all":
        query["subscription_tier"] = req.tier_filter

    # Fetch up to 10 000 recipients in batches using cursor
    recipients = await db.users.find(query, {"_id": 0, "email": 1, "name": 1}).to_list(10_000)
    if not recipients:
        return {"queued": 0, "message": "No matching recipients"}

    # Schedule the actual sends as a background task so the HTTP response is immediate
    background_tasks.add_task(_do_email_blast, req.subject, req.body_html, recipients)
    return {"queued": len(recipients), "message": f"Email blast queued for {len(recipients)} recipients"}

@api_router.get("/admin/newsletter-subscribers")
async def get_newsletter_subscribers(admin: dict = Depends(require_admin)):
    """Return all newsletter subscribers."""
    subs = await db.newsletter_subscribers.find(
        {"active": True}, {"_id": 0, "id": 1, "email": 1, "name": 1, "subscribed_at": 1}
    ).sort("subscribed_at", -1).to_list(5000)
    return {"count": len(subs), "subscribers": subs}

@api_router.get("/admin/contact-messages")
async def get_contact_messages(admin: dict = Depends(require_admin)):
    """Return all contact / support tickets."""
    tickets = await db.contact_messages.find(
        {}, {"_id": 0}
    ).sort("created_at", -1).to_list(500)
    return {"count": len(tickets), "tickets": tickets}


@api_router.get("/admin/reading-orders")
async def get_reading_orders(
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    admin: dict = Depends(require_admin),
):
    """Return all $19 personal-reading orders, newest first.

    Filterable by status (pending|paid|fulfilled). Pending rows are inserted
    by the buy-reading endpoint; the Stripe webhook flips them to paid; the
    admin marks them fulfilled when the reading is delivered."""
    limit = min(max(limit, 1), 200)
    skip = max(skip, 0)

    query: dict = {}
    if status:
        query["status"] = status

    orders = await db.reading_orders.find(
        query, {"_id": 0}
    ).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)

    total = await db.reading_orders.count_documents(query)

    # Counts by status (across all orders, ignoring the optional filter)
    counts_pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
    counts_rows = await db.reading_orders.aggregate(counts_pipeline).to_list(10)
    counts = {row["_id"] or "unknown": row["count"] for row in counts_rows}

    paid_total_cents = 0
    revenue_pipeline = [
        {"$match": {"status": {"$in": ["paid", "fulfilled"]}}},
        {"$group": {"_id": None, "total": {"$sum": "$amount_cents"}}},
    ]
    revenue_rows = await db.reading_orders.aggregate(revenue_pipeline).to_list(1)
    if revenue_rows:
        paid_total_cents = revenue_rows[0].get("total") or 0

    return {
        "orders": orders,
        "total": total,
        "skip": skip,
        "limit": limit,
        "counts": counts,
        "paid_total_cents": paid_total_cents,
    }


@api_router.put("/admin/reading-orders/{order_id}/status")
async def update_reading_order_status(
    order_id: str,
    status: str,
    admin: dict = Depends(require_admin),
):
    """Update the status of a reading order. Used to mark a paid order as
    fulfilled once the operator has delivered the written reading."""
    valid = {"pending", "paid", "fulfilled", "refunded"}
    if status not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {sorted(valid)}",
        )

    update = {"status": status}
    if status == "fulfilled":
        update["fulfilled_at"] = datetime.now(timezone.utc).isoformat()
        update["fulfilled_by"] = admin.get("id")

    result = await db.reading_orders.update_one(
        {"id": order_id},
        {"$set": update},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Reading order not found")

    return {"message": f"Order {order_id} updated to {status}"}

# Include router and setup middleware
app.include_router(api_router)

# Global exception handler: catch any unhandled exception, log it with a
# unique request ID for log correlation, and return a clean JSON 500 instead
# of FastAPI's default bare "Internal Server Error".
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())[:8]
    logging.error(
        "Unhandled exception [%s] %s %s: %s",
        request_id, request.method, request.url.path, exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred.", "request_id": request_id},
    )

# CORS: combining allow_credentials=True with allow_origins=["*"] is invalid per
# the CORS spec — browsers reject responses where Access-Control-Allow-Origin is
# the literal wildcard "*" and Access-Control-Allow-Credentials is "true".
# When the operator sets CORS_ORIGINS=* (the default for local dev), we use
# allow_origin_regex=".*" instead, which makes Starlette reflect the actual
# request Origin header in the response, satisfying the browser requirement.
_cors_origins = os.environ.get('CORS_ORIGINS', '*').split(',')
_cors_wildcard = '*' in _cors_origins
if _cors_wildcard:
    _frontend_url = os.environ.get('FRONTEND_URL', '')
    if _frontend_url.startswith('https://'):
        logging.warning(
            'SECURITY WARNING: CORS_ORIGINS is set to wildcard (*) but FRONTEND_URL '
            'suggests a production environment (%s). Set CORS_ORIGINS to your actual '
            'frontend domain(s) before going live.', _frontend_url
        )
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[] if _cors_wildcard else _cors_origins,
    allow_origin_regex='.*' if _cors_wildcard else None,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SecurityHeadersMiddleware)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
# Temporarily comment out entire create_indexes function to bypass all MongoDB index creation
# async def create_indexes():
#     """Create MongoDB indexes for performance"""
    # Users collection
    # Temporarily bypass MongoDB index creation to test SEO endpoints
# await db.users.create_index("email", unique=True)
    # Temporarily bypass remaining MongoDB index creation to test SEO endpoints
# await db.users.create_index("id", unique=True)
    # Temporarily bypass stripe_customer_id MongoDB index creation to unblock startup
# await db.users.create_index("stripe_customer_id", sparse=True)
    # Temporarily bypass password_reset_token MongoDB index creation to unblock startup
# await db.users.create_index("password_reset_token", sparse=True)
    # Chat messages
    # Temporarily bypass chat_messages.user_id MongoDB index creation to unblock startup
# await db.chat_messages.create_index("user_id")
    # Temporarily bypass chat_messages.session_id MongoDB index creation to unblock startup
# await db.chat_messages.create_index("session_id")
    # Temporarily bypass chat_messages compound MongoDB index creation to unblock startup
# await db.chat_messages.create_index([("user_id", 1), ("session_id", 1)])
    # Chat message daily limit helper index
    # Temporarily bypass chat_messages user_id/role/timestamp compound MongoDB index creation to unblock startup
# await db.chat_messages.create_index([("user_id", 1), ("role", 1), ("timestamp", 1)])
    # Friend messages (Saoul)
    # Temporarily bypass friend_messages.user_id MongoDB index creation to unblock startup
# await db.friend_messages.create_index("user_id")
    # Temporarily comment out orphaned friend_messages.session_id index creation to fix indentation
# await db.friend_messages.create_index("session_id")
    # Temporarily comment out orphaned friend_messages compound index creation to fix indentation
# await db.friend_messages.create_index([("user_id", 1), ("session_id", 1)])
    # Temporarily comment out orphaned friend_messages user_id/role/timestamp compound index creation to fix indentation
# await db.friend_messages.create_index([("user_id", 1), ("role", 1), ("timestamp", 1)])
    # Birth charts
    # Temporarily comment out orphaned birth_charts.user_id index creation to fix indentation
# await db.birth_charts.create_index("user_id", unique=True)
    # Temporarily comment out orphaned birth_charts.share_token index creation to fix indentation
# await db.birth_charts.create_index("share_token", sparse=True)
    # Compatibility reports
    # Temporarily comment out orphaned compatibility_reports.user_id index creation to fix indentation
# await db.compatibility_reports.create_index("user_id")
    # Temporarily comment out orphaned compatibility_reports.id index creation to fix indentation
# await db.compatibility_reports.create_index("id", unique=True)
    # Daily guidance cache
    # Temporarily comment out orphaned daily_guidance compound index creation to fix indentation
# await db.daily_guidance.create_index([("user_id", 1), ("date", 1)], unique=True)
    # Voice horoscope cache (binary MP3 per user per UTC day)
    # Temporarily comment out orphaned voice_horoscopes compound index creation to fix indentation
# await db.voice_horoscopes.create_index([("user_id", 1), ("date", 1)], unique=True)
    # Public per-sign horoscope cache (one row per sign per UTC day)
    # Temporarily comment out orphaned public_horoscopes compound index creation to fix indentation
# await db.public_horoscopes.create_index([("sign", 1), ("date", 1)], unique=True)
    # Public per-sign voice preview cache (binary MP3 per sign per UTC day)
    # Temporarily comment out orphaned voice_previews compound index creation to fix indentation
# await db.voice_previews.create_index([("sign", 1), ("date", 1)], unique=True)
    # Newsletter subscribers
    # Temporarily comment out orphaned newsletter_subscribers.email index creation to fix indentation
# await db.newsletter_subscribers.create_index("email", unique=True)
    # One-time personal reading orders
    # Temporarily comment out orphaned reading_orders.stripe_session_id index creation to fix indentation
# await db.reading_orders.create_index("stripe_session_id", unique=True, sparse=True)
    # Temporarily comment out orphaned reading_orders.user_id index creation to fix indentation
# await db.reading_orders.create_index("user_id", sparse=True)
    # Temporarily comment out orphaned reading_orders.email index creation to fix indentation
# await db.reading_orders.create_index("email", sparse=True)
    # Temporarily comment out orphaned reading_orders.created_at index creation to fix indentation
# await db.reading_orders.create_index("created_at")
    # Contact messages
    # Temporarily comment out orphaned contact_messages.created_at index creation to fix indentation
# await db.contact_messages.create_index("created_at")
    # Temporarily comment out orphaned MongoDB index creation logger line to fix indentation
# logger.info("MongoDB indexes created")

@app.on_event("shutdown")
async def shutdown_db_client():
    # client.close() (bypassed due to MongoDB service unavailability)
    pass
