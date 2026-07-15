# ER Diagram — Emotion Detection and Learning Support Engine

## Overview

This entity-relationship diagram describes the planned relational database schema for persisting user accounts and emotion-detection session records. The current application implementation uses CSV-based logging (`emotion_response_examples.csv`, `emotion_response_mapping.csv`) for simplicity and rapid development; this schema represents the structure a full MySQL backend would use for multi-user, persistent deployment.

## Entities

### USERS

| Field | Type | Key |
|---|---|---|
| email | VARCHAR | Primary Key (PK) |
| name | VARCHAR | |
| password | VARCHAR (hashed) | |
| role | VARCHAR | |
| login_count | INT | |
| created_at | DATETIME | |

### EMOTION_RECORDS

| Field | Type | Key |
|---|---|---|
| record_id | INT | Primary Key (PK) |
| email | VARCHAR | Foreign Key (FK) → Users.email |
| field | VARCHAR | |
| input_text | TEXT | |
| predicted_emotion | VARCHAR | |
| secondary_emotion | VARCHAR | |
| confidence_score | FLOAT | |
| model_used | VARCHAR | |
| ai_response | TEXT | |
| response_type | VARCHAR | |
| emotion_scores | JSON / TEXT | |
| timestamp | DATETIME | |
| csv_logged | BOOLEAN | |

## Relationship

**USERS (1) → (M) EMOTION_RECORDS**

- One User can generate many Emotion Records.
- Each Emotion Record belongs to exactly one User.
- `email` in `EMOTION_RECORDS` is a Foreign Key referencing `Users.email` (Primary Key).

## Notes

- This model stores emotion analysis sessions and AI-generated learning support responses.
- `predicted_emotion` and `secondary_emotion` correspond to the primary and mixed-emotion outputs shown in the application UI.
- `model_used` distinguishes which model (BiLSTM or BERT) produced a given record, supporting the app's side-by-side model comparison feature.
- `csv_logged` is a boolean flag indicating whether a record was also written to the CSV logs, supporting the transition period between file-based and database-backed storage.

## Diagram

![ER Diagram](ER_diagram.png)

## Legend

- **PK** — Primary Key
- **FK** — Foreign Key