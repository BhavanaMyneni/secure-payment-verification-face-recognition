"""
Verification endpoints.

POST /verify/face/   — accepts an image, runs face detection + matching against
                        the user's enrolled embedding, records the result.
POST /verify/otp/    — accepts a submitted OTP code, verifies it against the
                        user's TOTP secret. Only meaningful after a passed face check.

These are intentionally thin: the actual ML/OTP logic lives in src/, so it can
be tested and reused outside of Django (e.g. in a CLI or notebook).
"""

import json

import cv2
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from src.face_detection import FaceDetector
from src.face_recognition_model import FaceEmbedder, is_match
from src.otp_service import verify_otp

from .models import UserProfile, VerificationAttempt

_detector = FaceDetector()
_embedder = None  # lazy-loaded: requires the trained model file to exist


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = FaceEmbedder()
    return _embedder


@csrf_exempt
@require_POST
def verify_face(request):
    username = request.POST.get("username")
    image_file = request.FILES.get("image")

    if not username or not image_file:
        return JsonResponse({"error": "username and image are required"}, status=400)

    try:
        user = UserProfile.objects.get(username=username)
    except UserProfile.DoesNotExist:
        return JsonResponse({"error": "user not found"}, status=404)

    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    face = _detector.get_largest_face(frame)
    if face is None:
        return JsonResponse({"match": False, "reason": "no face detected"}, status=200)

    embedding = _get_embedder().embed(face)
    enrolled_embedding = np.array(user.face_embedding)
    matched = is_match(embedding, enrolled_embedding)

    VerificationAttempt.objects.create(user=user, face_match_passed=matched)
    return JsonResponse({"match": matched})


@csrf_exempt
@require_POST
def verify_otp_code(request):
    body = json.loads(request.body or "{}")
    username = body.get("username")
    code = body.get("otp")

    if not username or not code:
        return JsonResponse({"error": "username and otp are required"}, status=400)

    try:
        user = UserProfile.objects.get(username=username)
    except UserProfile.DoesNotExist:
        return JsonResponse({"error": "user not found"}, status=404)

    is_valid = verify_otp(user.otp_secret, code)

    latest_attempt = user.attempts.order_by("-timestamp").first()
    if latest_attempt:
        latest_attempt.otp_passed = is_valid
        latest_attempt.save(update_fields=["otp_passed"])

    return JsonResponse({"otp_verified": is_valid})
