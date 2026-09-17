"""
OTP (one-time password) generation and verification — the second authentication
factor, issued only after a successful face match.

Uses TOTP (time-based OTP, RFC 6238) via pyotp so each user's OTP secret can
be generated once at enrollment and stored securely (never in source code).
"""

import pyotp

from . import config


def generate_user_secret() -> str:
    """Call once per user at enrollment time; store the result in the user's DB record."""
    return pyotp.random_base32()


def generate_otp(user_secret: str) -> str:
    totp = pyotp.TOTP(user_secret, interval=config.OTP_VALIDITY_SECONDS)
    return totp.now()


def verify_otp(user_secret: str, submitted_code: str) -> bool:
    totp = pyotp.TOTP(user_secret, interval=config.OTP_VALIDITY_SECONDS)
    # valid_window=1 allows the previous/next OTP window to account for clock drift or delay
    return totp.verify(submitted_code, valid_window=1)
