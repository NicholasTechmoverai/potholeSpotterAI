#######################################
# 🚧 Pothole Spotter AI
#######################################
# Machine learning project that detects potholes 
# from images or live video streams using PyTorch.
#######################################

# 📂 Project Structure
automatedPatholeSpotter/
│-- app.py            # Main application script
│-- detect.py         # Detection logic
│-- my_model.pt       # Pre-trained model
│-- requirements.txt  # Dependencies
│-- .gitignore

#######################################
# 🔧 Setup & Usage
#######################################


# ASSUMING YOU HAVE PYTHON INSTALLED IF NOT DOWNLOAD IT FROM THIS LINK AND INSTALL IT 
https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe


# 1️⃣ Clone the repository
git clone https://github.com/NicholasTechmoverai/potholeSpotterAI.git
cd potholeSpotterAI

# 2️⃣ Create a virtual environment
# 👉 Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate

# 👉 Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the application
python app.py

#######################################
# 📝 Notes
#######################################
# - Use Python 3.8+ 
# - Always activate your .venv before running
# - Update dependencies if needed:
pip freeze > requirements.txt
