
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
            max_length=1000,
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
    try:
        data = request.json
        location = data.get("location")
        start_date = data.get("startDate")
        end_date = data.get("endDate")
        travel_type = data.get("travelType")

        # Validate inputs
        if not all([location, start_date, end_date, travel_type]):
            return jsonify({"error": "Missing required fields"}), 400

        # Parse dates
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        trip_duration = (end_date_obj - start_date_obj).days + 1

        if trip_duration <= 0:
            return jsonify({"error": "End date must be after start date"}), 400

        # Simplified query with clear formatting instructions
        query = (
            f"Generate a detailed {trip_duration}-day itinerary for a {travel_type} trip to {location} in a paragraph format for each day. "
            f"from {start_date} to {end_date}. Include daily activities and famous travel spots, in an explained paragraph"

        )

        # Get LLM response with specific parameters
        itinerary = llm.invoke(query, max_length=1000, temperature=0.7)
        
        # Process the response
        formatted_itinerary = "<div class='itinerary-container'>"
        
        # Split into days
        days = itinerary.split("Day")
        
        for i in range(1, trip_duration + 1):
            current_date = start_date_obj + timedelta(days=i-1)
            formatted_date = current_date.strftime("%B %d, %Y")
            
            # Find the corresponding day content
            day_content = ""
            for day in days:
                if day.strip().startswith(str(i)):
                    day_content = day.split(":", 1)[1].strip() if ":" in day else day.strip()
                    break
            
            # Format day container
            formatted_itinerary += f"""
                <div class='day-container'>
                    <h2 class='day-header'>Day {i}</h2>
                    <h3 class='date-header'>{formatted_date}</h3>
                    <div class='day-content'>
                        <p class='activity'>{day_content}</p>
                    </div>
                </div>
            """

        formatted_itinerary += "</div>"

        # CSS styling
        css_styles = """
            <style>
                .itinerary-container {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 20px auto;
                    padding: 20px;
                }
                .day-container {
                    margin-bottom: 30px;
                    padding: 20px;
                    border-radius: 8px;
                    background-color: #f9f9f9;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .day-header {
                    color: #2c3e50;
                    font-size: 24px;
                    margin: 0;
                    padding-bottom: 5px;
                    border-bottom: 2px solid #3498db;
                }
                .date-header {
                    color: #7f8c8d;
                    font-size: 18px;
                    margin: 10px 0;
                    font-weight: normal;
                }
                .day-content {
                    margin-top: 15px;
                }
                .activity {
                    margin: 10px 0;
                    line-height: 1.6;
                    color: #34495e;
                }
            </style>
        """

        final_output = css_styles + formatted_itinerary
        
        return jsonify({"itinerary": final_output})

    except Exception as e:
        print(f"Error generating itinerary: {str(e)}")
        return jsonify({
            "error": "Error generating itinerary",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
