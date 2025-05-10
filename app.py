from flask import Flask, request, jsonify
from flask_cors import CORS
import os, unicodedata
from dotenv import load_dotenv
from logs.log_handler import create_log_directory, get_log_directory, logs_recording
from service.process_service import (load_pdf_pages,  extract_experience_and_education_info,
                                     extract_personal_and_experience_info,transform_dict,
                                     normalize_text, start_faiss,research_faiss, api_chat)
import json
from flask import session



app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
CORS(app)

create_log_directory()
log_path = get_log_directory()
api_key = os.getenv("API_KEY")




@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:

        if 'pdf_file' not in request.files:
            error_message = 'Arquivo não encontrado'
            logs_recording(json.dumps({'error': error_message + '400'}), log_path)
            return jsonify({'error':  error_message }), 400

        pdf_file = request.files['pdf_file']

        if pdf_file.filename == '':
            error_message = "Nome de arquivo vazio"
            logs_recording(json.dumps({'error': error_message + '400'}), log_path)
            return jsonify({'error': error_message}), 400

        save_path = (f"./doc/{pdf_file.filename}")
        os.makedirs("./doc", exist_ok=True)
        pdf_file.save(save_path)

        pages = load_pdf_pages(save_path)
        personal_info = extract_personal_and_experience_info(pages)
        education_info = extract_experience_and_education_info(pages)

        return jsonify({
            "status": "sucesso",
            "response": {
                "mensagem": f'Arquivo {pdf_file.filename} recebido com sucesso!',
                "informacoes_pessoais_experiencia": personal_info,
                "experiencia_formacao": education_info
            }

        })
    except IndexError as e:
        error_message = f"{str(e)}"
        logs_recording(json.dumps({'error': error_message}), log_path)
        return jsonify({'error': error_message}), 400
    except Exception as e:
        error_message = "Erro inesperado no servidor"
        logs_recording(json.dumps({'error': error_message, 'exception': str(e)}), log_path)
        return jsonify({'error': error_message}), 500


@app.route('/carregar-dados', methods=['POST'])
def load_data():
    try:
        if not api_key:
            logs_recording(json.dumps({'error': 'Chave de API não encontrada no arquivo .env'}), log_path)
            return jsonify({'error': 'Chave de API ausente'}), 500
        else:
            print("Chave de API carregada com sucesso:")

        required_keys = ["experiencia_formacao", "informacoes_pessoais_experiencia"]

        data = request.get_json()

        missing = [k for k in required_keys if k not in data]

        if missing:
            error_message = {'error': f"Campos ausentes no JSON: {', '.join(missing)}, 400"}
            logs_recording(json.dumps(error_message), log_path)
            return jsonify(error_message), 400

        data_dict = transform_dict(data)
        normalized_data = {key: normalize_text(value) for key, value in data_dict.items()}

        db = start_faiss(api_key, normalized_data)
        session["faiss_loaded"] = True

        return jsonify({
            "status": "sucesso",
            "response": {
                "mensagem": f'DB FAISS iniciou com sucesso!',
                "informacoes_pessoais_experiencia": {
                    "informacoes_pessoais": normalized_data.get("informacoes_pessoais", ""),
                    "experiencia_profissional": normalized_data.get("experiencia_profissional", "")
                },
                "experiencia_formacao": {
                    "formacao": normalized_data.get("formacao", "")
                }
            }
        })

    except Exception as e:
        error_message = "Erro inesperado no servidor"
        logs_recording(json.dumps({'error': error_message, 'exception': str(e)}), log_path)
        return jsonify({'error': error_message}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not session.get("faiss_loaded"):
            logs_recording(json.dumps({'error': 'Índice FAISS ainda não carregado para esta sessão'}), log_path)
            return jsonify({"error": "Índice FAISS ainda não carregado para esta sessão."}), 400

        data = request.get_json()

        required_fields = ["pergunta", "filtro"]
        missing = [field for field in required_fields if
                   field not in data or not isinstance(data[field], str) or not data[field].strip()]

        if missing:
            error_message = {'error': f"Campos obrigatórios ausentes ou inválidos: {', '.join(missing)}", "code": 400}
            logs_recording(json.dumps(error_message), log_path)
            return jsonify(error_message), 400

        question = data["pergunta"].strip()
        filter_to_db = data["filtro"].strip()

        valid_filters = ["formacao", "informacoes_pessoais", "experiencia_profissional"]
        if filter_to_db not in valid_filters:
            return jsonify({"error": f"Filtro inválido. Use um dos seguintes: {', '.join(valid_filters)}"}), 400

        answer_db = research_faiss(api_key, filter_to_db, question)

        response_chat = api_chat(question, api_key, answer_db)

        return jsonify({
            "status": "sucesso",
            "mensagem": response_chat
        })

    except Exception as e:
        error_message = "Erro inesperado no servidor"
        logs_recording(json.dumps({'error': error_message, 'exception': str(e)}), log_path)
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)