from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for, send_from_directory
from PyPDF2 import PdfFileReader, PdfFileWriter
from googletrans import Translator
from werkzeug.utils import secure_filename
import os

app = Flask(__name__, static_url_path="/static")


# Configuration

app.config['UPLOAD_FOLDER'] = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'
app.config['TRANSLATED_FOLDER'] = os.path.dirname(os.path.abspath(__file__)) + '/translated/'

translator = Translator()

def translate_pdf(input_path, output_path, target_language='en'):
    try:
        input_file = PdfFileReader(open(input_path, 'rb'))
        output = PdfFileWriter()

        translated_content = ""

        for page_number in range(input_file.getNumPages()):
            page = input_file.getPage(page_number)
            text = page.extractText()
            translated_text = translator.translate(text, dest=target_language)
            translated_content += translated_text.text

            translated_page_writer = open(output_path, 'ab')
            translated_page_writer.write(translated_text.text.encode('utf-8'))
            translated_page_writer.close()

        return True, translated_content
    except Exception as e:
        print(f"Translation failed: {str(e)}")
        return False, None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    
    if request == 'POST':
        print("papapapppa")
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            translated_filename = f"translated_{filename}"
            translated_output_path = os.path.join(app.config['TRANSLATED_FOLDER'], translated_filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            success, translated_content = translate_pdf(input_path, translated_output_path, target_language='fr')

            if success:
                return redirect(url_for('uploaded_file', filename=filename))
            #jsonify({'success': 'File uploaded and translated successfully', 'translated_content': translated_content})
            else:
                return jsonify({'error': 'Translation failed'})
            
    return render_template('index.html')


@app.route('/translated/', methods=['GET'])
def download_file(filename):
    return send_from_directory(os.path.join(app.config['TRANSLATED_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 1000))
    app.run(host='0.0.0.0', port=port)
