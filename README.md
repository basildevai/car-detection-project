# Car Detection Project

A web-based application for car classification and object detection, built for the Capstone Project: Computer Vision.

## Folder Structure
- `backend/`: Flask API and models
  - `app.py`: Flask server
  - `fine_tuned_mobilenetv2.h5`: Classification model
  - `mobilenet_bbox_model.h5`: Object detection model
  - `class_names.json`: Class ID to name mapping
  - `requirements.txt`: Backend dependencies
- `frontend/`: React frontend
  - `index.html`: Single-page HTML with React and Tailwind
- `README.md`: This file

## Setup Instructions
### Backend
1. Navigate to `backend/`:

   ```bash
   cd backend

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows

3. Install dependencies:

   ```bash
   pip install -r requirements.txt

4. Ensure `fine_tuned_mobilenetv2.h5`, `mobilenet_bbox_model.h5`, and `class_names.json` are in `backend/`.

5. Run the Flask server

   ```bash
   python app.py

  The API will be available at http://localhost:5000.

### Frontend
1. Navigate to `frontend/`:

   ```bash
   cd frontend

2. Open index.html in a browser (e.g., via a local server or directly):

   ```bash
   python -m http.server 8000

    Access at `http://localhost:8000`.

3. Ensure the backend is running to handle API requests.

## Deployment to Production

### Backend (e.g., Heroku)

1. Install Heroku CLI and log in:

   ```bash
    heroku login

2. Create a Heroku app:

   ```bash
    heroku create car-detection-api

3. Add a `Procfile` in `backend/`:

    ```text
    web: gunicorn app:app

4. Install gunicorn:

   ```bash
    pip install gunicorn
    pip freeze > requirements.txt

5. Deploy to Heroku:

   ```bash
    git init
    git add .
    git commit -m "Initial commit"
    heroku git:remote -a car-detection-api
    git push heroku main

6. Update the frontend's fetch URL to the Heroku app URL (e.g., https://car-detection-api.herokuapp.com/predict).