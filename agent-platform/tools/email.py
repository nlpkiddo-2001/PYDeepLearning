"""
Email Tool
==========
Send emails via SMTP. Credentials are read from environment variables.
"""

from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from tools.registry import tool


@tool(
    name="send_email",
    description="Send an email to the specified recipient with a subject and body.",
    tags=["email", "notification"],
)
def send_email(to: str, subject: str, body: str, html: bool = False) -> str:
    """Send an email via SMTP."""
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")

    if not smtp_user or not smtp_pass:
        return "ERROR: SMTP_USER and SMTP_PASS environment variables must be set."

    msg = MIMEMultipart("alternative")
    msg["From"] = smtp_user
    msg["To"] = to
    msg["Subject"] = subject

    content_type = "html" if html else "plain"
    msg.attach(MIMEText(body, content_type))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [to], msg.as_string())
        return f"Email sent successfully to {to}"
    except Exception as exc:
        return f"ERROR sending email: {exc}"
