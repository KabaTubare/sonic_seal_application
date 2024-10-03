import sys
import asyncio
import aiofiles
from aiohttp import ClientSession
import shutil
import numpy as np
import os
sys.path.insert(0, '/app/audioseal/src')
import torch
import torchaudio
import tempfile
import random
import string
from datetime import datetime
from audioseal import AudioSeal
import json
import socket
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import logging
from flask import Flask, request, jsonify, send_file, render_template, make_response, Response
from cryptography.hazmat.primitives import serialization
import subprocess
import io
import matplotlib.pyplot as plt
from PIL import Image
import torchaudio.transforms as T
from flask import url_for
from base64 import b64encode, b64decode
from datetime import timedelta

app = Flask(__name__)

@app.route('/watermark', methods=['POST'])
async def upload_and_watermark_route():
    logging.debug("Received watermark request")
    return await audio_watermarking_app.upload_and_watermark()

def check_port_availability(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

if check_port_availability('0.0.0.0', 8080):
    print("Port 8080 is already in use. Please ensure the port is free before starting the application.")
else:
    print("Port 8080 is available. Starting the application.")

# Setup basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def safe_load_audio(audio_file_path):
    try:
        waveform, sample_rate = torchaudio.load(audio_file_path)
        if waveform.size(0) > 1:  # Convert to mono if needed
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, sample_rate
    except RuntimeError as e:
        logging.exception(f"Loading error, attempting conversion: {e}")
        gcs_path = await upload_for_conversion(audio_file_path)
        logging.info(f"Uploaded for conversion: {gcs_path}")
        converted_file_name = os.path.basename(audio_file_path).rsplit('.', 1)[0] + "_converted.wav"
        local_converted_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        await download_converted(converted_file_name, local_converted_path)
        logging.info(f"Downloaded converted file: {local_converted_path}")
        return torchaudio.load(local_converted_path)
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        raise e

def convert_audio_format(audio_stream):
    try:
        temp_wav_path = "/tmp/temp_audio.wav"
        with open(temp_wav_path, "wb") as out_file:
            out_file.write(audio_stream.getbuffer())
        result = subprocess.run(["ffmpeg", "-i", temp_wav_path, temp_wav_path], capture_output=True, text=True)
        if result.returncode == 0:
            waveform, sample_rate = torchaudio.load(temp_wav_path)
            return waveform, sample_rate
        else:
            logging.error(f"FFmpeg conversion failed with error: {result.stderr}")
            raise RuntimeError(f"Audio format conversion failed: {result.stderr}")
    except Exception as e:
        logging.error(f"An error occurred during audio format conversion: {e}")
        raise

def extract_mfcc_features(waveform, sample_rate):
    try:
        # Setup MFCC transformation parameters
        mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=40, melkwargs={'n_fft': 2048, 'n_mels': 128, 'hop_length': 512, 'win_length': 2048})
        # Apply the MFCC transformation
        mfcc = mfcc_transform(waveform)
        return mfcc
    except Exception as e:
        # Log the exception with a custom message and the exception details
        logging.error(f"Failed to extract MFCC features: {e}", exc_info=True)
        raise RuntimeError(f"Failed to extract MFCC features: {e}")

def plot_mfcc(mfcc):
    # Ensure MFCC data is 2D by squeezing or selecting appropriate dimensions
    if mfcc.ndim == 3:
        mfcc = mfcc.squeeze(0)

    if not isinstance(mfcc, np.ndarray):
        mfcc = mfcc.detach().numpy()  # Ensure it's a numpy array for plotting

    if mfcc.ndim != 2:
        raise ValueError("MFCC data should be 2D after processing.")

    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc, cmap='viridis', aspect='auto', origin='lower')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    return buf.getvalue()  # Correctly return the byte stream

def plot_spectrogram(waveform, sample_rate):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    spectrogram_transform = T.Spectrogram(n_fft=2048, win_length=None, hop_length=512, center=True)
    spectrogram = spectrogram_transform(waveform)
    spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_db[0].cpu().numpy(), cmap='hot', aspect='auto', origin='lower')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def ensure_directories_exist():
    directories = ['static/spectrograms']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

ensure_directories_exist()

def waveform_to_base64_spectrogram(waveform, sample_rate):
    spectrogram_image = plot_spectrogram(waveform, sample_rate)
    buf = io.BytesIO()
    spectrogram_image.save(buf, format='PNG')
    base64_string = b64encode(buf.getvalue()).decode('utf-8')
    return base64_string

class AudioWatermarkingApp:
    def __init__(self, app):
        self.app = app
        self.metadata_file = 'audio_metadata.json'
        self.private_key, self.public_key = self.generate_keys()
        self.public_key_path = os.environ.get('PUBLIC_KEY_PATH')
        self.generator = AudioSeal.load_generator("audioseal_wm_16bits")
        self.detector = AudioSeal.load_detector("audioseal_detector_16bits")
        if os.environ.get('USE_FILE_KEYS', 'false').lower() == 'true':
            with open(self.public_key_path, 'wb') as key_file:
                key_file.write(self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
        self.ensure_metadata_file_exists()

    def detect_immediate_watermark(self, audio_file_path):
        """Detect the watermark immediately after watermarking using AudioSeal's detector."""
        try:
            # Load and resample the audio
            waveform, sample_rate = self.load_and_resample_audio(audio_file_path, 16000)
            logging.debug(f"Audio loaded and resampled to 16 kHz, sample rate: {sample_rate}")

            # Load the detector
            detector = AudioSeal.load_detector("audioseal_detector_16bits")
            logging.debug("AudioSeal detector model loaded successfully")

            # Perform the watermark detection
            detection_result, message_tensor = detector.detect_watermark(waveform.unsqueeze(0), 16000)
            confidence = float(detection_result)
            logging.debug(f"Watermark detection completed with confidence: {confidence}")

            if confidence < 0.5:
                logging.debug("Confidence level below threshold, marking as Not Watermarked")
                return {
                    "detection_result": "Not Watermarked",
                    "confidence": confidence,
                    "detected_message": None,
                    "match": "No",
                    "spectrogram_image_url": None
                }

            # Convert the detected binary message to hexadecimal
            detected_message = self.binary_to_message(''.join(str(bit) for bit in message_tensor[0].tolist()))
            detected_message_trimmed = detected_message[:4].lower()
            logging.debug(f"Detected binary message converted to hex: {detected_message}")
            logging.debug(f"Detected message (first 4 characters): {detected_message_trimmed}")

            # Prepare the response
            response = {
                "detection_result": "Watermarked" if confidence >= 0.5 else "Not Watermarked",
                "confidence": confidence,
                "detected_message": detected_message_trimmed,
            }

            # Log the final response
            logging.debug(f"Response: {response}")

            # Return the response directly without external modification
            return response

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {"error": str(e)}

    def extract_mfcc_features(self, waveform, sample_rate):
        """Extract MFCC features from the audio waveform."""
        try:
            # Setup MFCC transformation parameters
            mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=40, melkwargs={'n_fft': 2048, 'n_mels': 128, 'hop_length': 512, 'win_length': 2048})
            # Apply the MFCC transformation
            mfcc = mfcc_transform(waveform)
            logging.debug("Extracted MFCC features successfully")
            return mfcc
        except Exception as e:
            # Log the exception with a custom message and the exception details
            logging.error(f"Failed to extract MFCC features: {e}", exc_info=True)
            raise RuntimeError(f"Failed to extract MFCC features: {e}")

    def plot_mfcc(self, mfcc):
        """Plot MFCC features and return the image as a byte stream."""
        # Ensure MFCC data is 2D by squeezing or selecting appropriate dimensions
        if mfcc.ndim == 3:
            mfcc = mfcc.squeeze(0)

        if not isinstance(mfcc, np.ndarray):
            mfcc = mfcc.detach().numpy()  # Ensure it's a numpy array for plotting

        if mfcc.ndim != 2:
            raise ValueError("MFCC data should be 2D after processing.")

        plt.figure(figsize=(10, 4))
        plt.imshow(mfcc, cmap='viridis', aspect='auto', origin='lower')
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        return buf.getvalue()  # Correctly return the byte stream

    def ensure_metadata_file_exists(self):
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)

    def plot_spectrogram(self, waveform, sample_rate):
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        spectrogram_transform = T.Spectrogram(n_fft=2048, win_length=None, hop_length=512, center=True)
        spectrogram = spectrogram_transform(waveform)
        spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram_db[0].cpu().numpy(), cmap='hot', aspect='auto', origin='lower')
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        return Image.open(buf)

    def prepare_and_store_user_message(self, input_message):
        binary_str = ''.join(format(ord(c), '08b') for c in input_message)
        hex_str = self.binary_to_message(binary_str)
        self.save_audio_metadata(input_message, hex_str)
        return hex_str

    def message_to_binary(self, message, bit_length=16):
        if not message:
            raise ValueError("The input message is empty and cannot be converted to binary.")
        binary_message = ''.join(format(ord(c), '08b') for c in message)
        # Ensure binary_message has the required length and pad with zeros if necessary
        binary_message = binary_message.ljust(bit_length, '0')[:bit_length]
        logger.debug(f"Converted message to binary: {binary_message}")
        return [int(binary_message[i]) for i in range(len(binary_message))]

    def binary_to_message(self, binary_str):
        try:
            logging.debug(f"Received binary string for conversion: {binary_str}")
            if len(binary_str) % 8 != 0:
                padding_length = 8 - (len(binary_str) % 8)
                binary_str = '0' * padding_length + binary_str
                logging.debug(f"Binary string after padding: {binary_str}")
            byte_array = int(binary_str, 2).to_bytes((len(binary_str) // 8), byteorder='big')
            hex_string = byte_array.hex()
            logging.debug(f"Hexadecimal representation: {hex_string}")
            return hex_string
        except Exception as e:
            logging.error("Failed to convert binary to hex due to: ", exc_info=True)
            return None

    def generate_keys(self):
        private_key = rsa.generate_private_key(backend=default_backend(), public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        return private_key, public_key

    def sign_message(self, message):
        signature = self.private_key.sign(
            message.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return signature

    def verify_signature(self, message, signature):
        try:
            decoded_signature = bytes.fromhex(signature)
            self.public_key.verify(
                decoded_signature,
                message.encode(),
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"Unexpected exception in verify_signature: {e}")
            return False

    async def upload_and_watermark(self):
        file_keys = [key for key in request.files if key.startswith('audio')]
        unique_messages = [request.form.get(f'unique_message{key.split("audio")[1]}') for key in file_keys]

        tasks = []
        for index, (file_key, unique_message) in enumerate(zip(file_keys, unique_messages)):
            audio_file = request.files[file_key]
            original_audio_name = f"original_{audio_file.filename}"
            original_audio_path = os.path.join('static', 'audio', original_audio_name)
            os.makedirs(os.path.dirname(original_audio_path), exist_ok=True)
            tasks.append(self.process_and_watermark_file(audio_file, original_audio_path, unique_message, index + 1))

        responses = await asyncio.gather(*tasks)
        return jsonify(responses)

    async def process_and_watermark_file(self, audio_file, original_audio_path, unique_message, index):
        async with aiofiles.open(original_audio_path, 'wb') as out_file:
            content = audio_file.read()  # Note: no await here
            await out_file.write(content)
        logger.info(f'Original audio file saved at: {original_audio_path}')

        try:
        # Load the audio file and extract necessary features
            waveform, sample_rate = await safe_load_audio(original_audio_path)
            if waveform is None or sample_rate is None:
                logger.error("Waveform or sample rate not loaded properly.")
                raise ValueError("Waveform or sample rate not loaded properly.")

        # Generate spectrogram and MFCC for the original audio
            pre_watermark_spectrogram = waveform_to_base64_spectrogram(waveform, sample_rate)
            pre_watermark_mfcc = self.extract_mfcc_features(waveform, sample_rate)
            pre_watermark_mfcc_img = self.plot_mfcc(pre_watermark_mfcc)
            pre_watermark_mfcc_encoded = b64encode(pre_watermark_mfcc_img).decode('utf-8')

        # Apply the watermark and get the path of the watermarked audio file
            watermarked_audio_path, hex_message, signature_hex = await self.watermark_audio(original_audio_path, unique_message)

        # Move the watermarked audio file to the static directory
            watermarked_audio_filename = os.path.basename(watermarked_audio_path)
            static_watermarked_audio_path = os.path.join('static', 'audio', watermarked_audio_filename)
            shutil.move(watermarked_audio_path, static_watermarked_audio_path)
            logger.info(f'Watermarked audio file saved at: {static_watermarked_audio_path}')

        # Load the watermarked audio file and extract necessary features
            watermarked_waveform, _ = await safe_load_audio(static_watermarked_audio_path)
            post_watermark_spectrogram = waveform_to_base64_spectrogram(watermarked_waveform, sample_rate)
            post_watermark_mfcc = self.extract_mfcc_features(watermarked_waveform, sample_rate)
            post_watermark_mfcc_img = self.plot_mfcc(post_watermark_mfcc)
            post_watermark_mfcc_encoded = b64encode(post_watermark_mfcc_img).decode('utf-8')

        # Detect the watermark in the watermarked audio file
            detected_watermark = await self.detect_immediate_watermark(static_watermarked_audio_path)
            detected_message = detected_watermark.get('detected_message', '')

        # Prepare the response to send back to the client
            response = {
                'filename': f'Audio {index}: {audio_file.filename}',
                'download_url': url_for('static', filename=f'audio/{watermarked_audio_filename}'),
                'hex_message': hex_message,
                'signature_hex': signature_hex,
                'pre_watermark_spectrogram': pre_watermark_spectrogram,
                'post_watermark_spectrogram': post_watermark_spectrogram,
                'pre_watermark_mfcc': pre_watermark_mfcc_encoded,
                'post_watermark_mfcc': post_watermark_mfcc_encoded,
                'pre_watermark_audio_url': url_for('static', filename=f'audio/{os.path.basename(original_audio_path)}'),
                'post_watermark_audio_url': url_for('static', filename=f'audio/{watermarked_audio_filename}'),
                'pre_watermark_spectrogram_label': 'Spectrogram Before Watermarking',
                'post_watermark_spectrogram_label': 'Spectrogram After Watermarking',
                'pre_watermark_mfcc_label': 'MFCC Before Watermarking',
                'post_watermark_mfcc_label': 'MFCC After Watermarking',
                'detection_result': detected_watermark.get('detection_result', 'Error'),
                'confidence': detected_watermark.get('confidence', 0),
                'detected_message': detected_message,
                'match': detected_watermark.get('match', "N/A"),
                'spectrogram_image_url': detected_watermark.get('spectrogram_image_url', '')
            }
            logger.info(f'Response generated: {response}')
            return response

        except Exception as e:
            logger.exception(f'An error occurred during watermarking: {e}')
            return {'error': str(e)}, 500

    async def watermark_audio(self, audio_file_path, unique_message):
        """Apply a watermark to the audio file using AudioSeal's generator."""
        try:
            waveform, sample_rate = await self.load_and_resample_audio(audio_file_path, 16000)
            logging.debug(f"Loaded and resampled audio: {audio_file_path} to 16 kHz")

            generator = AudioSeal.load_generator("audioseal_wm_16bits")
            logging.debug("AudioSeal generator loaded successfully")

            if unique_message:
                binary_indices = self.message_to_binary(unique_message)
                message_tensor = torch.tensor(binary_indices, dtype=torch.int32).unsqueeze(0)
                watermark = generator.get_watermark(waveform.unsqueeze(0), 16000, message=message_tensor).squeeze(0)
                logging.debug(f"Watermark applied using unique message: {unique_message}")
            else:
                watermark = generator.get_watermark(waveform.unsqueeze(0), 16000).squeeze(0)
                logging.debug("Default watermark applied")

            watermarked_audio = waveform + watermark

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
            torchaudio.save(temp_file, watermarked_audio, 16000)
            logging.debug(f"Watermarked audio saved to temporary file: {temp_file}")

            signature = self.sign_message(unique_message)
            signature_hex = signature.hex()
            logging.debug(f"Digital signature created: {signature_hex}")

            hex_message = self.message_to_hex(unique_message)
            self.save_audio_metadata(unique_message, signature_hex)

            return temp_file, hex_message, signature_hex

        except Exception as e:
            logger.error(f"Error in watermarking audio: {e}")
            raise RuntimeError("Failed to watermark audio.") from e

    async def detect_watermark(self, audio_file_path, original_message=None):
        """Detect the watermark in an audio file using AudioSeal's detector."""
        try:
            # Load and resample the audio
            waveform, sample_rate = await self.load_and_resample_audio(audio_file_path, 16000)
            logging.debug(f"Audio loaded and resampled to 16 kHz, sample rate: {sample_rate}")

            # Load the detector
            detector = AudioSeal.load_detector("audioseal_detector_16bits")
            logging.debug("AudioSeal detector model loaded successfully")

            # Perform the watermark detection
            detection_result = detector.detect_watermark(waveform.unsqueeze(0), 16000)
            if asyncio.iscoroutine(detection_result):
                detection_result = await detection_result

            confidence, message_tensor = detection_result
            confidence = float(confidence)
            logging.debug(f"Watermark detection completed with confidence: {confidence}")

            if confidence < 0.5:
                logging.debug("Confidence level below threshold, marking as Not Watermarked")
                return {
                    "detection_result": "Not Watermarked",
                    "confidence": confidence,
                    "detected_message": None,
                    "match": "No",
                    "spectrogram_image_url": None
                }

            # Convert the detected binary message to hexadecimal
            detected_message = self.binary_to_message(''.join(str(bit) for bit in message_tensor[0].tolist()))
            detected_message_trimmed = detected_message[:4].lower()
            logging.debug(f"Detected binary message converted to hex: {detected_message}")
            logging.debug(f"Detected message (first 4 characters): {detected_message_trimmed}")

            # Optionally verify the detected message against the original message
            match = "N/A"
            if original_message:
                original_hex = self.message_to_hex(original_message)
                match = "Yes" if original_hex.startswith(detected_message_trimmed) else "No"

            # Prepare the response
            response = {
                "detection_result": "Watermarked" if confidence >= 0.5 else "Not Watermarked",
                "confidence": confidence,
                "detected_message": detected_message_trimmed,
                "match": match,
                "spectrogram_image_url": None  # You can add the spectrogram URL if needed
            }

            # Log the final response
            logging.debug(f"Response: {response}")

            # Return the response directly without external modification
            return response

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {"error": str(e)}


    @app.route('/detect', methods=['POST'])
    async def detect_watermark_route():
        try:
            if 'audio' not in request.files:
                logger.debug("Audio file is missing in the request.")
                return jsonify({'error': 'Audio file is required.'}), 400

            audio_file = request.files['audio']
            original_message = request.form.get('original_message', None)

            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
            logger.debug(f"Temporary file created at {temp_path}")

        # Read the audio file content synchronously
            content = audio_file.read()
            logger.debug(f"Read {len(content)} bytes from the uploaded audio file")

        # Write the content to the temporary file asynchronously
            async with aiofiles.open(temp_path, 'wb') as out_file:
                await out_file.write(content)
            logger.debug(f"Written audio content to temporary file {temp_path}")

            detection_response = await audio_watermarking_app.detect_watermark(temp_path, original_message)
            logger.debug(f"Detection response: {detection_response}")

            os.remove(temp_path)
            logger.debug(f"Temporary file {temp_path} removed")

            response = {
                'detection_result': detection_response.get('detection_result', 'Error'),
                'confidence': detection_response.get('confidence', 0),
                'detected_message': detection_response.get('detected_message', ''),
                'match': detection_response.get('match', "N/A"),
                'input_message': original_message if original_message else "Not provided",
                'expected_hex_message': detection_response.get('expected_hex_message', "Not applicable"),
                'spectrogram_image_url': detection_response.get('spectrogram_image_url', '')
            }

            logger.debug(f"Response to be sent: {response}")
            return jsonify(response)

        except Exception as e:
            logger.error(f"Error during detection route: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/')
    def home():
        return render_template('index.html')

    def generate_unique_message(self, length=16):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))
        logging.debug(f"Generated unique message: {unique_message}")

    def message_to_binary(self, message, bit_length=16):
        """Converts a string message to a binary representation fit for tensor operations."""
        if not message:
            logger.error("The input message is empty and cannot be converted to binary.")
            raise ValueError("The input message is empty and cannot be converted to binary.")
        binary_message = ''.join(format(ord(c), '08b') for c in message)
        # Ensure binary_message has the required length and pad with zeros if necessary
        binary_message = binary_message.ljust(bit_length, '0')[:bit_length]
        logger.debug(f"Converted message to binary: {binary_message}")
        return [int(binary_message[i]) for i in range(len(binary_message))]

    def binary_to_message(self, binary_str):
        try:
            logging.debug(f"Received binary string for conversion: {binary_str}")
            if len(binary_str) % 8 != 0:
                padding_length = 8 - (len(binary_str) % 8)
                binary_str = '0' * padding_length + binary_str
                logging.debug(f"Binary string after padding: {binary_str}")
            byte_array = int(binary_str, 2).to_bytes((len(binary_str) // 8), byteorder='big')
            hex_string = byte_array.hex()
            logging.debug(f"Hexadecimal representation: {hex_string}")
            return hex_string
        except Exception as e:
            logging.error("Failed to convert binary to hex due to: ", exc_info=True)
            return None

    async def load_and_resample_audio(self, audio_file_path, target_sample_rate=16000):
        waveform, sample_rate = await safe_load_audio(audio_file_path)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        logging.debug(f"Resampled audio from {sample_rate} to {target_sample_rate}")
        return waveform, target_sample_rate

    def message_to_hex(self, message):
        if message is None:
            logger.error("The input message cannot be None.")
            raise ValueError("The input message cannot be None.")
        return ''.join(format(ord(c), '02x') for c in message)

    def get_metadata(self):
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                logging.debug(f"Metadata loaded successfully from {self.metadata_file}")
                return metadata
        except FileNotFoundError:
            logging.warning(f"Metadata file {self.metadata_file} not found, returning empty dictionary")
            return {}

    def save_metadata(self, metadata):
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
                logging.debug(f"Metadata saved successfully to {self.metadata_file}")
        except Exception as e:
            logging.error(f"Failed to save metadata: {e}", exc_info=True)

    def save_audio_metadata(self, message, signature_hex):
        try:
            hex_message = self.message_to_hex(message)
            with open(self.metadata_file, 'r+') as f:
                data = json.load(f)
                logging.debug(f"Loaded existing metadata for update: {data}")
                data['audio_files'] = data.get('audio_files', {})
                data['audio_files'][hex_message] = {'signature': signature_hex}
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
                logging.debug(f"Updated metadata with new audio data under key {hex_message}")
        except Exception as e:
            logging.error(f"Failed to update audio metadata: {e}", exc_info=True)

    async def watermark_audio(self, audio_file_path, unique_message):
        """Apply a watermark to the audio file using AudioSeal's generator and immediately detect it."""
        try:
            waveform, sample_rate = await self.load_and_resample_audio(audio_file_path, 16000)
            logging.debug(f"Loaded and resampled audio: {audio_file_path} to 16 kHz")

            generator = AudioSeal.load_generator("audioseal_wm_16bits")
            logging.debug("AudioSeal generator loaded successfully")

            if unique_message:
                binary_indices = self.message_to_binary(unique_message)
                message_tensor = torch.tensor(binary_indices, dtype=torch.int32).unsqueeze(0)
                watermark = generator.get_watermark(waveform.unsqueeze(0), 16000, message=message_tensor).squeeze(0)
                logging.debug(f"Watermark applied using unique message: {unique_message}")
            else:
                watermark = generator.get_watermark(waveform.unsqueeze(0), 16000).squeeze(0)
                logging.debug("Default watermark applied")

            watermarked_audio = waveform + watermark

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
            torchaudio.save(temp_file, watermarked_audio, 16000)
            logging.debug(f"Watermarked audio saved to temporary file: {temp_file}")

        # Immediately detect the watermark to ensure correctness
            detection_result = await self.detect_immediate_watermark(temp_file)
            detected_message = detection_result.get('detected_message', '')

            if detection_result.get('detection_result') != 'Watermarked':
                raise RuntimeError("Watermark detection failed immediately after watermarking.")

            signature = self.sign_message(unique_message)
            signature_hex = signature.hex()
            logging.debug(f"Digital signature created: {signature_hex}")

            hex_message = detected_message  # Use the detected message as the hex to ensure correctness
            self.save_audio_metadata(unique_message, signature_hex)

            return temp_file, hex_message, signature_hex

        except Exception as e:
            logger.error(f"Error in watermarking audio: {e}")
            raise RuntimeError("Failed to watermark audio.") from e
        
    async def detect_immediate_watermark(self, audio_file_path):
        """Detect the watermark immediately after watermarking using AudioSeal's detector."""
        try:
        # Load and resample the audio
            waveform, sample_rate = await self.load_and_resample_audio(audio_file_path, 16000)
            logging.debug(f"Audio loaded and resampled to 16 kHz, sample rate: {sample_rate}")

        # Load the detector
            detector = AudioSeal.load_detector("audioseal_detector_16bits")
            logging.debug("AudioSeal detector model loaded successfully")

        # Perform the watermark detection
            detection_result = detector.detect_watermark(waveform.unsqueeze(0), 16000)
        
        # If detect_watermark returns a coroutine, await it
            if asyncio.iscoroutine(detection_result):
                detection_result = await detection_result

            confidence, message_tensor = detection_result
            confidence = float(confidence)
            logging.debug(f"Watermark detection completed with confidence: {confidence}")

            if confidence < 0.5:
                logging.debug("Confidence level below threshold, marking as Not Watermarked")
                return {
                    "detection_result": "Not Watermarked",
                    "confidence": confidence,
                    "detected_message": None,
                    "match": "No",
                    "spectrogram_image_url": None
                }

        # Convert the detected binary message to hexadecimal
            detected_message = self.binary_to_message(''.join(str(bit) for bit in message_tensor[0].tolist()))
            detected_message_trimmed = detected_message[:4].lower()
            logging.debug(f"Detected binary message converted to hex: {detected_message}")
            logging.debug(f"Detected message (first 4 characters): {detected_message_trimmed}")

        # Prepare the response
            response = {
                "detection_result": "Watermarked" if confidence >= 0.5 else "Not Watermarked",
                "confidence": confidence,
                "detected_message": detected_message_trimmed,
            }

        # Log the final response
            logging.debug(f"Response: {response}")

        # Return the response directly without external modification
            return response

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {"error": str(e)}
        
audio_watermarking_app = AudioWatermarkingApp(app)

if __name__ == '__main__':
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(24)
    app.config['UPLOAD_FOLDER'] = '/path/to/upload/folder'
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
    ensure_directories_exist()
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
