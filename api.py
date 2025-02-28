from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import sqlite3
from datetime import datetime
import base64
from pathlib import Path
from openai import OpenAI
from flask_cors import CORS

# Importar funciones de app.py
from app import (
    analyze_query_intent, ask_database, create_visualization,
    query_documents, process_documents, setup_document_directory,
    get_api_key, get_database_info, generate_database_schema_string,
    get_table_names, extract_document_context
)

# Configuración
DOCUMENT_DIR = "docs"
CHROMA_DIR = "chroma_db"
STATIC_DIR = "static" 
VISUALIZATIONS_DIR = os.path.join(STATIC_DIR, "visualizations")

# Crear directorios necesarios
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Conexión a la base de datos con check_same_thread=False para usar en Flask
conn = sqlite3.connect('data/flight_radar4.db', check_same_thread=False)



app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)  # Habilitar CORS para todas las rutas

# Por esta configuración más específica
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Variables globales para el estado
openai_client = None
doc_db = None
conversation_history = []

# Añade esto a api.py
@app.route('/api/ping', methods=['GET'])
def ping():
    return jsonify({"status": "success", "message": "API is running"})


@app.route('/api/chat', methods=['POST'])
def chat():
    global conversation_history, openai_client, doc_db
    
    data = request.json
    user_message = data.get('message', '')
    
    # Añadir pregunta del usuario al historial
    conversation_history.append({"role": "user", "content": user_message})
    
    # Analizar la intención del usuario
    query_intent = analyze_query_intent(user_message)
    
    # Determinar qué herramientas usar basado en la intención
    needs_db_query = query_intent["needs_db_query"]
    needs_date_formatting = query_intent.get("needs_date_formatting", False)
    
    needs_visualization = any(keyword in user_message.lower() for keyword in 
                            ['gráfico', 'visualiza', 'grafica', 'mostrar', 'ver', 'plot', 'chart',
                             'diagrama', 'barras', 'líneas', 'pastel', 'dispersión', 'histograma'])
    
    needs_document_query = any(keyword in user_message.lower() for keyword in 
                             ['documento', 'pdf', 'archivo', 'texto', 'buscar en', 'encontrar en',
                              'información sobre', 'qué dice', 'menciona', 'contiene'])
    
    # Preparar herramientas según necesidades
    tools = []
    
    if needs_db_query:
        db_schema = generate_database_schema_string()
        tools.append({
            "type": "function",
            "function": {
                "name": "ask_database",
                "description": "Use this function to answer user questions about the database by executing SQL queries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                                SQL query to extract information from the database.
                                SQL should be written using this database schema:
                                {db_schema}
                                The query should be specific and directly address the user's question.
                                Always use SELECT statements only. Never use DROP, DELETE, UPDATE, or ALTER statements.
                                Limit your results to avoid excessive data.
                            """
                        }
                    },
                    "required": ["query"]
                }
            }
        })
    
    if needs_visualization:
        tools.append({
            "type": "function",
            "function": {
                "name": "create_visualization",
                "description": "Use this function to create data visualizations based on query results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "object",
                            "description": "The results object returned from the ask_database function. Should contain the query results to visualize."
                        },
                        "visualization_type": {
                            "type": "string",
                            "description": "The type of visualization to create.",
                            "enum": ["bar", "line", "pie", "scatter", "heatmap", "histogram"]
                        },
                        "title": {
                            "type": "string",
                            "description": "The title for the visualization."
                        },
                        "x_label": {
                            "type": "string",
                            "description": "The label for the x-axis."
                        },
                        "y_label": {
                            "type": "string",
                            "description": "The label for the y-axis."
                        },
                        "categorical": {
                            "type": "boolean",
                            "description": "Whether the data is categorical (true) or numerical (false).",
                            "default": False
                        },
                        "chart_style": {
                            "type": "string",
                            "description": "The style for the chart.",
                            "enum": ["darkgrid", "whitegrid", "dark", "white", "ticks"],
                            "default": "darkgrid"
                        },
                        "color_palette": {
                            "type": "string",
                            "description": "The color palette to use for the visualization.",
                            "enum": ["viridis", "plasma", "inferno", "magma", "cividis", "mako", "rocket", "Blues", "Greens", "Reds", "Purples", "Oranges"],
                            "default": "viridis"
                        }
                    },
                    "required": ["results", "visualization_type", "title"]
                }
            }
        })
    
    if needs_document_query and doc_db is not None:
        tools.append({
            "type": "function",
            "function": {
                "name": "query_documents",
                "description": "Use this function to search for relevant information in the uploaded documents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant document sections."
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of document chunks to retrieve.",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        })
    
    try:
        # Si no tenemos cliente OpenAI, crear una respuesta simulada para pruebas
        if openai_client is None:
            # Respuesta simulada para pruebas sin OpenAI
            if needs_db_query:
                assistant_response = "He analizado tu consulta sobre la base de datos. Para responderla necesitaría ejecutar una consulta SQL. ¿Quieres que proceda con esto?"
            elif needs_visualization:
                assistant_response = "Entiendo que quieres visualizar datos. ¿Qué tipo de gráfico prefieres ver?"
            elif needs_document_query:
                assistant_response = "He detectado que quieres buscar información en los documentos. ¿Podrías especificar más lo que buscas?"
            else:
                assistant_response = "Hola, soy un asistente virtual especializado en análisis de datos. ¿En qué puedo ayudarte hoy?"
                
            conversation_history.append({"role": "assistant", "content": assistant_response})
            return jsonify({"response": assistant_response})
        
        # Si tenemos cliente OpenAI, usarlo normalmente
        chat_completion = openai_client.chat.completions.create(
            messages=conversation_history,
            model="gpt-3.5-turbo",
            tools=tools if tools else None,
            temperature=0.2
        )
        
        assistant_message = chat_completion.choices[0].message
        
        # Verificar si la respuesta incluye llamadas a herramientas
        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
            # Añadir el mensaje del asistente al historial
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    } for tool_call in assistant_message.tool_calls
                ]
            })
            
            visualization_url = None
            
            # Procesar cada llamada a herramienta
            for tool_call in assistant_message.tool_calls:
                try:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == 'ask_database':
                        query = function_args["query"]
                        print(f"Ejecutando consulta SQL: {query}")
                        
                        function_response = ask_database(query)
                        
                    elif function_name == 'create_visualization':
                        print(f"Creando visualización: {function_args['visualization_type']}")
                        
                        graphic, error = create_visualization(**function_args)
                        if graphic:
                            # Guardar la visualización y devolver una URL
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"viz_{timestamp}.png"
                            file_path = os.path.join(VISUALIZATIONS_DIR, filename)
                            
                            with open(file_path, "wb") as f:
                                f.write(base64.b64decode(graphic))
                                
                            visualization_url = f"/api/visualizations/{filename}"
                            function_response = {
                                "success": True, 
                                "message": "Visualización creada correctamente",
                                "visualization_url": visualization_url
                            }
                        else:
                            function_response = {"success": False, "error": error}
                            
                    elif function_name == 'query_documents':
                        query = function_args["query"]
                        n_results = function_args.get("n_results", 5)
                        print(f"Buscando en documentos: {query}")
                        
                        # Obtener documentos relevantes
                        relevant_docs = query_documents(doc_db, query, n_results)
                        
                        # Extraer contexto de los documentos
                        document_context = extract_document_context(relevant_docs)
                        
                        if document_context:
                            function_response = {
                                "success": True,
                                "documents_found": len(relevant_docs),
                                "context": document_context
                            }
                        else:
                            function_response = {
                                "success": False,
                                "message": "No se encontraron documentos relevantes."
                            }
                    
                    # Añadir la respuesta de la herramienta al historial
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(function_response, ensure_ascii=False)
                    })
                    
                except Exception as e:
                    error_message = f"Error ejecutando {tool_call.function.name}: {str(e)}"
                    print(f"ERROR: {error_message}")
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps({"error": error_message})
                    })
            
            # Obtener la respuesta final del asistente
            final_response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversation_history,
                temperature=0.2
            )
            
            assistant_response = final_response.choices[0].message.content
            
            # Añadir la respuesta final al historial
            conversation_history.append({"role": "assistant", "content": assistant_response})
            
            response_data = {
                "response": assistant_response,
                "conversation_history": [msg for msg in conversation_history if msg["role"] in ["user", "assistant"]][-10:]
            }
            
            if visualization_url:
                response_data["visualization_url"] = visualization_url
                
            return jsonify(response_data)
            
        else:
            # Si no hay llamadas a herramientas, usamos la respuesta directamente
            assistant_response = assistant_message.content
            
            # Añadir la respuesta final al historial
            conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return jsonify({
                "response": assistant_response,
                "conversation_history": [msg for msg in conversation_history if msg["role"] in ["user", "assistant"]][-10:]
            })
        
    except Exception as e:
        error_message = f"Error en la comunicación con OpenAI: {str(e)}"
        print(f"ERROR: {error_message}")
        conversation_history.append({"role": "assistant", "content": f"Lo siento, ocurrió un error: {error_message}"})
        return jsonify({"error": error_message, "response": "Lo siento, ha ocurrido un error al procesar tu mensaje."}), 500

@app.route('/api/visualizations/<filename>', methods=['GET'])
def get_visualization(filename):
    """Obtiene una imagen de visualización guardada"""
    return send_from_directory(VISUALIZATIONS_DIR, filename)

@app.route('/api/clear-chat', methods=['POST'])
def clear_chat():
    """Limpia el historial de conversación"""
    global conversation_history
    
    # Mantener solo el mensaje del sistema
    if conversation_history and conversation_history[0]["role"] == "system":
        conversation_history = [conversation_history[0]]
    else:
        conversation_history = []
        
    return jsonify({"status": "success", "message": "Chat history cleared"})

def initialize_server():
    global openai_client, doc_db, conversation_history
    
    print("Inicializando servidor...")
    
    try:
        # Configuración de OpenAI
        api_key = get_api_key()
        print(f"API Key obtenida: {'Sí' if api_key else 'No'}")
        
        if api_key:
            try:
                openai_client = OpenAI(api_key=api_key)
                print("✅ Cliente OpenAI inicializado correctamente")
            except Exception as e:
                print(f"❌ Error al inicializar OpenAI: {str(e)}")
                openai_client = None
        else:
            print("⚠️ No se pudo obtener la API key de OpenAI.")
            openai_client = None
            
        # Procesar documentos
        doc_db, processed_files = process_documents(api_key)
        if doc_db:
            print(f"✅ Base de datos vectorial iniciada con {len(processed_files)} documentos")
        else:
            print("⚠️ No se pudo inicializar la base de datos vectorial")
            
        # Obtener información detallada de la base de datos
        detailed_schema = generate_database_schema_string()
        
        # Configurar el mensaje del sistema
        conversation_history = [
            {
                "role": "system",
                "content": f"""Eres un asistente virtual avanzado especializado en consultas de bases de datos y análisis de datos.
                
                INFORMACIÓN DE LA BASE DE DATOS:
                {detailed_schema}
                
                CAPACIDADES PRINCIPALES:
                - Consultas SQL avanzadas con optimización de rendimiento
                - Visualización de datos dinámica y adaptativa
                - Análisis semántico de documentos y correlación con datos estructurados
                - Análisis multidimensional y estadísticas predictivas
                
                Cuando no estés seguro de la respuesta o los datos solicitados no estén disponibles, indícalo claramente y ofrece alternativas.
                """
            }
        ]
        
        print("✅ Servidor inicializado y listo para recibir solicitudes")
        
    except Exception as e:
        print(f"❌ Error durante la inicialización: {str(e)}")

if __name__ == '__main__':
    initialize_server()
    app.run(debug=True, host='0.0.0.0', port=5001) 