from io import BytesIO
import os
from flask import Flask, render_template, request, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from Translator-API/src/utils.py import translate_pdf


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)


class Upload(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	filename = db.Column(db.String(50))
	data = db.Column(db.LargeBinary)
	
class Translated(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	main_file = db.ForeignKey('Upload')
	filename = db.Column(db.String(50))
	data = db.Column(db.LargeBinary)



@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		file = request.files['file']
		
        translated_file = translate_pdf(file, target_language='fr')
        tranlated = Translated(filename=f"translated_{file.filename}", data=translated_file.read())


		upload = Upload(filename=file.filename, data=file.read())
		
        db.session.add(tranlated)
		db.session.add(upload)
		db.session.commit()
		return f'Uploaded: {file.filename} and the translated version has been saved.'
	return render_template('index.html')


@app.route('/download/<translated_id>')
def download(translated_id):
	translated = Translated.query.filter_by(id=translated_id).first()
	return send_file(BytesIO(translated.data), download_name=translated.filename, as_attachment=True )



if __name__ == '__main__':
	
    port = int(os.environ.get("PORT", 1000))
    app.run(host='0.0.0.0', port=port)
	