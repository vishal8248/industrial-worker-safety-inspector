import os
import smtplib
from email.message import EmailMessage


class EmailNotifier:
    def __init__(self):
        self.smtp_host = os.getenv(
            "SMTP_HOST",
            "smtp.gmail.com",
        )

        self.smtp_port = int(
            os.getenv(
                "SMTP_PORT",
                "587",
            )
        )

        self.sender_email = os.getenv(
            "ALERT_SENDER_EMAIL"
        )

        self.sender_password = os.getenv(
            "ALERT_SENDER_PASSWORD"
        )

        self.receiver_email = os.getenv(
            "ALERT_RECEIVER_EMAIL"
        )

        if not self.sender_email:
            raise RuntimeError(
                "ALERT_SENDER_EMAIL is not configured."
            )

        if not self.sender_password:
            raise RuntimeError(
                "ALERT_SENDER_PASSWORD is not configured."
            )

        if not self.receiver_email:
            raise RuntimeError(
                "ALERT_RECEIVER_EMAIL is not configured."
            )

    def send_incident_alert(
        self,
        subject: str,
        report: str,
    ):
        message = EmailMessage()

        message["Subject"] = subject
        message["From"] = self.sender_email
        message["To"] = self.receiver_email

        message.set_content(report)

        with smtplib.SMTP(
            self.smtp_host,
            self.smtp_port,
        ) as server:
            server.starttls()

            server.login(
                self.sender_email,
                self.sender_password,
            )

            server.send_message(
                message
            )