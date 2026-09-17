"""
Django models for the verification app.

NOTE: face embeddings and OTP secrets are sensitive — in production these
should be encrypted at rest, not stored as plain fields. This is a starting
structure to build on, not a production-ready security implementation.
"""

from django.db import models


class UserProfile(models.Model):
    username = models.CharField(max_length=150, unique=True)
    email = models.EmailField(unique=True)

    # Stored as a serialized vector (e.g. JSON-encoded list of floats) captured at enrollment.
    face_embedding = models.JSONField()

    # Per-user TOTP secret, generated once at enrollment via otp_service.generate_user_secret().
    otp_secret = models.CharField(max_length=64)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username


class VerificationAttempt(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="attempts")
    face_match_passed = models.BooleanField(default=False)
    otp_passed = models.BooleanField(default=False)
    timestamp = models.DateTimeField(auto_now_add=True)

    @property
    def fully_verified(self) -> bool:
        return self.face_match_passed and self.otp_passed
