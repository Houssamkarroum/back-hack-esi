from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import google.generativeai as genai
from deep_translator import GoogleTranslator
from PIL import Image
from opencage.geocoder import OpenCageGeocode
import requests
import os
import tempfile
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Configure Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = os.getenv("FLASK_SECRET_KEY", 'dev-secret-key')

# Configure APIs
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
geocoder = OpenCageGeocode(os.getenv("OPENCAGE_API_KEY"))

# Helper functions
def translate_text(text, target_lang='ar'):
    try:
        return GoogleTranslator(source='auto', target='ar').translate(text)
    except Exception as e:
        print(f"Translation failed: {str(e)}")
        return text

@app.route('/api/analyze-image', methods=['POST'])
def api_analyze_image():
    if 'file' not in request.files:
        return jsonify({
            "error": "No file uploaded",
            "translation": translate_text("No file uploaded", request.form.get('lang', 'ar'))
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            "error": "No file selected",
            "translation": translate_text("No file selected", request.form.get('lang', 'ar'))
        }), 400
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            file.save(temp_file.name)
            img = Image.open(temp_file.name)
            
            # Analyze image
            response = model.generate_content([
                "Analyze this medical image and describe any medical conditions it might show. "
                "Provide a detailed medical analysis including possible diagnoses, "
                "recommended next steps, and when to consult a doctor.", 
                "responded with very simple text not md text.", 
                img
            ])
            
            diagnosis = response.text
            lang = request.form.get('lang', 'ar')
            
            return jsonify({
                "diagnosis": diagnosis,
                "translation": translate_text(diagnosis, lang),
                "type": "diagnosis"
            })
            
    except Exception as e:
        return jsonify({
            "error": str(e),
            "translation": translate_text("An error occurred during image analysis", request.form.get('lang', 'ar'))
        }), 500
        
    finally:
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file.name)
            except:
                pass

@app.route('/api/medication-advice', methods=['POST'])
def api_medication_advice():
    data = request.get_json()
    if not data or 'symptoms' not in data:
        return jsonify({
            "error": "Symptoms description required",
            "translation": translate_text("Symptoms description required", data.get('lang', 'ar') if data else 'ar')
        }), 400
    
    try:
        lang = data.get('lang', 'ar')
        response = model.generate_content(
            f"Given the following symptoms: {data['symptoms']}, provide possible medications or treatments that might help. "
            "Specify over-the-counter and prescription options if appropriate. Also, note when to consult a doctor."
        )
        
        advice = response.text
        
        return jsonify({
            "advice": advice,
            "translation": translate_text(advice, lang),
            "type": "medication"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "translation": translate_text("An error occurred during analysis", data.get('lang', 'ar') if data else 'ar')
        }), 500

@app.route('/api/find-specialist', methods=['POST'])
def api_find_specialist():
    data = request.get_json()
    if not data or 'illness' not in data:
        return jsonify({
            "error": "Illness description required",
            "translation": translate_text("Illness description required", data.get('lang', 'ar') if data else 'ar')
        }), 400
    
    try:
        lang = data.get('lang', 'ar')
        response = model.generate_content(
            f"Based on the illness or condition: {data['illness']}, suggest the type of medical specialist that the person should consult."
        )
        
        specialist = response.text
        
        return jsonify({
            "specialist": specialist,
            "translation": translate_text(specialist, lang),
            "type": "specialist"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "translation": translate_text("An error occurred while finding specialist", data.get('lang', 'ar') if data else 'ar')
        }), 500

@app.route('/api/find-hospitals', methods=['OPTIONS', 'POST'])
def api_find_hospitals():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    data = request.get_json()
    if not data or 'location' not in data:
        return jsonify({
            "error": "Location required",
            "translation": translate_text("Location required", data.get('lang', 'ar') if data else 'ar')
        }), 400
    
    try:
        result = geocoder.geocode(data['location'])
        if not result:
            return jsonify({
                "error": "Could not find location",
                "translation": translate_text("Could not find location", data.get('lang', 'ar'))
            }), 400
        
        latitude = result[0]['geometry']['lat']
        longitude = result[0]['geometry']['lng']
        
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
        (
          node["amenity"="hospital"](around:5000,{latitude},{longitude});
          node["amenity"="clinic"](around:5000,{latitude},{longitude});
          node["healthcare"="doctor"](around:5000,{latitude},{longitude});
        );
        out center;
        """
        
        response = requests.post(overpass_url, data=overpass_query)
        response_data = response.json()
        
        facilities = []
        for element in response_data.get('elements', []):
            name = element['tags'].get('name', 'Unnamed Facility')
            lat = element.get('lat')
            lon = element.get('lon')
            if lat and lon:
                maps_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
                facilities.append({
                    'name': translate_text(name, data.get('lang', 'ar') if data else 'ar'),
                    'maps_link': maps_link,
                    'latitude': lat,
                    'longitude': lon
                })
        
        return jsonify({
            "location": data['location'],
            "facilities": facilities,
            "count": len(facilities)
        })
        
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Overpass API request failed",
            "translation": translate_text("Map service unavailable", data.get('lang', 'ar') if data else 'ar')
        }), 503
    except Exception as e:
        return jsonify({
            "error": str(e),
            "translation": translate_text("An error occurred while searching for hospitals", data.get('lang', 'ar') if data else 'ar')
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)