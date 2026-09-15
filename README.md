# Industrial Worker Safety Inspector

An AI-powered industrial workplace safety monitoring system that combines
YOLO-based worker tracking, safety-zone monitoring, Vision-Language Model (VLM)
analysis, RAG-based SOP retrieval, and LangGraph workflows to identify and
report potential safety incidents.

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
     |                      |
     v                      v
Incident Report       Supervisor Email