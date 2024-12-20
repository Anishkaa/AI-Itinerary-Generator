
import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from datetime import datetime, timedelta

load_dotenv()
app = Flask(__name__)

# HuggingFace LLM configuration
HF_TOKEN = os.getenv("HF_TOKEN")  # Ensure this environment variable is set
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set")

# Initialize LLM with error handling
def init_llm():
    try:
        return HuggingFaceEndpoint(
            repo_id="tiiuae/falcon-7b-instruct",
            token=HF_TOKEN,
            max_length=800,
            temperature=0.7
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

llm = init_llm()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate-itinerary", methods=["POST"])
def generate_itinerary():
    data = request.json
    location = data.get("location")
    start_date = data.get("startDate")
    end_date = data.get("endDate")
    travel_type = data.get("travelType")

    # Validate inputs
    if not location or not start_date or not end_date or not travel_type:
        return jsonify({"error": "Missing required fields"}), 400

    # Calculate the number of days for the trip
    try:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        trip_duration = (end_date_obj - start_date_obj).days + 1  # Include both start and end dates

        if trip_duration <= 0:
            return jsonify({"error": "End date must be after start date"}), 400
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    # Generate a query for the LLM
    query = (
        f"Generate a detailed {trip_duration}-day itinerary for a {travel_type} trip to {location} in a paragraph format for each day. "
        f"from {start_date} to {end_date}. Include daily activities and famous travel spots, in an explained paragraph"
    )

    try:
        # Call the LLM and get the response
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        num_days = (end_date_obj - start_date_obj).days + 1

        query = f"Generate a detailed {num_days}-day itinerary for {travel_type} in {location} from {start_date} to {end_date}."
        itinerary = llm.invoke(query)
        itinerary = llm.invoke(query)
        days = itinerary.split("Day")  # Split by the keyword "Day" to identify each day's content

        formatted_itinerary = ""
        for i, day in enumerate(days):
            if i == 0:
                continue  # Skip the first part as it doesn't have "Day"
            
            # Rebuild the header for each day
            day_header = day.split(":")[0].strip()  # e.g., "1, 04 Dec, 2024"
            activities = day.split(":")[1].strip() if len(day.split(":")) > 1 else ""  # Get the activities for that day
            
            # Replace newlines in the activities with <p> tags to create separate paragraphs
            activities_paragraphs = activities.split('\n')
            formatted_activities = ''.join([f"<p>{activity.strip()}</p>" for activity in activities_paragraphs if activity.strip()])
            
            # Format the output: Bold the day header and ensure activities appear below as separate paragraphs
            formatted_itinerary += f"<b>Day {day_header}:</b><br>{formatted_activities}<br>"

        return jsonify({"itinerary": formatted_itinerary})# Return the formatted itinerary as a JSON response

    except Exception as e:
        return jsonify({"error": f"Error generating itinerary: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
