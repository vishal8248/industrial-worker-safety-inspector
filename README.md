# Industrial Worker Safety Inspector

An AI-powered industrial workplace safety monitoring system that combines
YOLO-based worker tracking, zone monitoring, VLM-based incident understanding,
RAG-based SOP retrieval, and LangGraph workflows to identify and report
potential safety incidents.

## Overview

Industrial safety monitoring systems often rely on simple object detection,
which can identify that a worker is present but cannot understand the context
of a potential safety incident.

This project combines computer vision and Generative AI to create a
context-aware safety inspection pipeline.

The system:

- Detects and tracks workers using YOLO + ByteTrack
- Monitors configured restricted safety zones
- Measures worker dwell time inside hazardous zones
- Uses a Vision-Language Model (VLM) to analyze incident context
- Applies predefined safety rules to confirm incidents
- Retrieves relevant safety procedures using RAG
- Uses LangGraph to manage the incident workflow
- Generates an evidence-based incident report
- Sends an email alert to the responsible supervisor

## Architecture

```text
CCTV / Video
     |
     v
YOLO + ByteTrack
     |
     v
Worker Detection & Tracking
     |
     v
Worker Foot Point
     |
     v
Configured Safety Zone (Kavach)
     |
     v
Zone Entry / Dwell Monitoring
     |
     v
Incident Candidate
     |
     v
Vision-Language Model
     |
     v
Contextual Safety Observations
     |
     v
Safety Rules
     |
     v
Confirmed Incident
     |
     v
RAG
     |
     v
Relevant Safety SOP
     |
     v
LangGraph Incident Workflow
     |
     +----------------------+
     v                      v
Incident Report       Supervisor Email

## Future Improvements

The current system uses fixed-camera video and image-space safety zones.
A production-grade system could be extended with additional sensing,
visualization, and monitoring capabilities.

Potential improvements include:

- **LiDAR / Depth Sensors** — Add depth information for more accurate
  distance measurement, 3D worker positioning, and spatial safety-zone
  monitoring.

- **Safety Monitoring Dashboard** — Build a web UI to display live camera
  feeds, active workers, safety-zone status, detected incidents, and system
  health.

- **Real-Time CCTV Integration** — Connect the pipeline to live industrial
  CCTV/IP camera streams instead of pre-recorded videos.

- **Multi-Camera Correlation** — Correlate worker and incident information
  across multiple cameras covering different areas of a facility.

- **Incident Severity Scoring** — Assign severity levels based on incident
  type, duration, location, and available evidence.

- **Incident Evidence Clips** — Automatically save a short video segment
  before and after a confirmed incident.

- **Additional Notification Channels** — Support notifications through
  channels such as SMS, messaging platforms, or enterprise alert systems.

- **Historical Safety Analytics** — Store incidents and provide dashboards
  for identifying recurring safety issues and high-risk areas.

- **Improved PPE Analysis** — Use specialized computer-vision models and
  higher-resolution camera feeds for more reliable PPE verification.

- **Production Deployment & Monitoring** — Add containerization, service
  monitoring, logging, model versioning, and scalable deployment for
  industrial environments.