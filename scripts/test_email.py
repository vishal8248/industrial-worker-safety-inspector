from dotenv import load_dotenv

from app.notifications.email import EmailNotifier


def main():
    load_dotenv()

    notifier = EmailNotifier()

    notifier.send_incident_alert(
        subject="Safety Incident Test",
        report=(
            "INDUSTRIAL SAFETY INCIDENT REPORT\n\n"
            "Incident Type: phone_usage\n"
            "Worker ID: 2\n"
            "Camera: camera_01\n"
            "Timestamp: 10.07s\n\n"
            "Evidence:\n"
            "- Worker is visibly using a mobile phone."
        ),
    )

    print(
        "Test incident email sent successfully."
    )


if __name__ == "__main__":
    main()