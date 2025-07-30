ASL Alphabet Image Classifier
This project is a Convolutional Neural Network (CNN) built with TensorFlow to classify static images of American Sign Language (ASL) alphabet letters.

Features
Classifies ASL hand signs (A–Z, excluding J and Z since they require motion)

Trained on the ASL Alphabet dataset

Achieves high accuracy on unseen test images

Provides a foundation for future real-time ASL recognition on a Raspberry Pi using OpenCV and MediaPipe

Tech Stack
Language: Python 3

Framework: TensorFlow / Keras

Libraries: NumPy, Matplotlib, OpenCV (for future use)

Hardware Used
Training was done on a desktop with:

Intel i7‑13700K

Project Structure
sql
Copy
Edit
ASL-Classifier/
│-- data/               # ASL dataset (not included in repo)
│-- models/             # Saved trained model
│-- notebooks/          # Jupyter Notebooks for training and testing
│-- src/                # Training and inference scripts
│-- README.md
Results
Final test accuracy: X% (fill this in)

Example predictions: (add screenshots or tables later)

How to Run
Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/asl-classifier.git
cd asl-classifier
Install dependencies:

nginx
Copy
Edit
pip install -r requirements.txt
Train the model:

nginx
Copy
Edit
python train.py
Run inference on test images:

arduino
Copy
Edit
python predict.py --image path_to_image.jpg
Next Steps
Add real-time detection using OpenCV and MediaPipe

Deploy on Raspberry Pi for live ASL recognition

Dataset
The project uses the ASL Alphabet dataset from Kaggle:
https://www.kaggle.com/grassknoted/asl-alphabet
