import sqlite3
from openai import OpenAI
import os
import json
from datetime import datetime
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
from flask import g

# Dependencias para análisis de documentos
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Conexión a la base de datos
#Modificado # conn = sqlite3.connect('data/flight_radar4.db')

# Configuración para análisis de documentos
DOCUMENT_DIR = "docs"
CHROMA_DIR = "chroma_db"
DATABASE = 'data/flight_radar4.db'

import sqlite3
from openai import OpenAI
import os
import json
from datetime import datetime
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
from flask import g

# Dependencias para análisis de documentos
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Configuración para análisis de documentos
DOCUMENT_DIR = "docs"
CHROMA_DIR = "chroma_db"
DATABASE = 'data/flight_radar4.db'


def init_db(app):
    """
    Initialize database-related configurations for the Flask app.
    """
    # Register the close_db function to run at the end of each request
    app.teardown_appcontext(close_db)
    
    # Asegurarse de que existe el directorio de datos
    import os
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
    
    # Inicializar g._database aquí no es necesario porque
    # get_db() se encargará de eso cuando se llame
def get_db():
    """
    Get a database connection.
    Creates a new connection if one doesn't exist for this context.
    """
    from flask import g
    # Check if a database connection already exists in the application context
    db = getattr(g, '_database', None)
    if db is None:
        # Create a new connection if it doesn't exist
        # Use check_same_thread=False only when using Flask's thread-local storage
        db = g._database = sqlite3.connect(DATABASE, check_same_thread=False)
    return db

def close_db(error=None):
    """
    Close the database connection at the end of the request.
    """
    from flask import g
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()



def get_table_names():
    """Obtener todos los nombres de tablas de la base de datos."""
    table_names = []
    db = get_db()
    tables = db.execute('SELECT name FROM sqlite_master WHERE type="table"')
    for table in tables.fetchall():
        table_names.append(table[0])
    return table_names

def get_column_names(table_name):
    """Obtener los nombres de columnas para una tabla específica."""
    column_names = []
    db = get_db()
    columns = db.execute(f"PRAGMA table_info('{table_name}')")
    for col in columns:
        column_names.append(col[1])
    return column_names

def get_table_sample(table_name, limit=5):
    """Obtener una muestra de datos de una tabla para entender su estructura."""
    try:
        db = get_db()
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        sample_data = db.execute(query).fetchall()
        columns = get_column_names(table_name)
        
        # Formatear los datos para mejor visualización
        formatted_data = []
        for row in sample_data:
            formatted_data.append(dict(zip(columns, row)))
            
        return formatted_data
    except Exception as e:
        return f"Error al obtener muestra de la tabla {table_name}: {e}"

def get_database_info():
    """Obtener información detallada sobre la estructura de la base de datos."""
    database_info = []
    tables = get_table_names()
    
    for table_name in tables:
        columns = get_column_names(table_name)
        sample_data = get_table_sample(table_name, limit=2)
        
        # Contar filas
        try:
            db = get_db()
            row_count = db.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        except Exception as e:
            row_count = f"Error al contar filas: {str(e)}"
            
        # Obtener información sobre columnas (tipos de datos)
        column_info = []
        for col in columns:
            try:
                # Detectar tipo de datos basado en muestra
                data_type = "desconocido"
                if isinstance(sample_data, list) and len(sample_data) > 0:
                    if col in sample_data[0]:
                        value = sample_data[0][col]
                        if value is not None:
                            data_type = type(value).__name__
                
                column_info.append({
                    "name": col,
                    "data_type": data_type
                })
            except Exception as e:
                column_info.append({
                    "name": col,
                    "data_type": f"error al detectar: {str(e)}"
                })
                
        database_info.append({
            "table_name": table_name,
            "columns": column_info,
            "row_count": row_count,
            "sample_data": sample_data if not isinstance(sample_data, str) else "Error: " + sample_data
        })
    
    return database_info

def generate_database_schema_string():
    """Generar una representación de texto del esquema de la base de datos."""
    db_info = get_database_info()
    schema_parts = []
    
    for table in db_info:
        table_name = table["table_name"]
        columns_info = [f"{col['name']} ({col['data_type']})" for col in table["columns"]]
        row_count = table["row_count"]
        
        schema_parts.append(f"Tabla: {table_name} ({row_count} filas)")
        schema_parts.append(f"Columnas: {', '.join(columns_info)}")
        
        # Añadir muestra de datos si disponible
        if isinstance(table["sample_data"], list) and len(table["sample_data"]) > 0:
            sample_str = "Ejemplo de datos:\n"
            for i, row in enumerate(table["sample_data"][:2]):
                sample_str += f"  Fila {i+1}: {json.dumps(row, ensure_ascii=False)}\n"
            schema_parts.append(sample_str)
        
        schema_parts.append("")  # Línea en blanco para separar tablas
        
    return "\n".join(schema_parts)

def analyze_query_intent(user_question):
    """Analiza la intención del usuario para determinar si necesita consulta a base de datos."""
    # Palabras clave relacionadas con bases de datos
    db_keywords = [
        'base de datos', 'consulta', 'sql', 'tabla', 'datos', 'registros', 
        'seleccionar', 'mostrar', 'cuántos', 'promedio', 'máximo', 'mínimo',
        'buscar', 'encontrar', 'listar', 'contar'
    ]
    
    # Añadir nombres de tablas y columnas como palabras clave
    tables = get_table_names()
    all_keywords = db_keywords + tables
    
    for table in tables:
        columns = get_column_names(table)
        all_keywords.extend(columns)
    
    # Normalizar palabras clave y pregunta para comparación
    def normalize(text):
        return text.lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    
    normalized_question = normalize(user_question)
    normalized_keywords = [normalize(kw) for kw in all_keywords]
    
    # Verificar coincidencias
    matches = []
    for keyword in normalized_keywords:
        if keyword in normalized_question:
            matches.append(keyword)
    
    # Verificar si el usuario solicita información de fecha/hora
    date_related_keywords = [
        'fecha', 'tiempo', 'cuando', 'día', 'mes', 'año', 'hora',
        'date', 'time', 'when', 'day', 'month', 'year'
    ]
    
    needs_date_formatting = any(keyword in normalized_question for keyword in 
                             [normalize(k) for k in date_related_keywords])
    
    # Determinar si es probable que se necesite consulta
    needs_db_query = len(matches) > 0
    
    return {
        "needs_db_query": needs_db_query,
        "matched_keywords": matches,
        "tables": tables,
        "question": user_question,
        "needs_date_formatting": needs_date_formatting
    }

def modify_query_for_date_formatting(query, table_names):
    """
    Modifica la consulta SQL para formatear automáticamente columnas de fecha.
    
    Args:
        query: La consulta SQL original
        table_names: Nombres de las tablas en la base de datos
    
    Returns:
        La consulta modificada con conversión de fechas
    """
    # Obtener información de todas las columnas de todas las tablas
    all_columns = {}
    date_columns = []
    
    for table in table_names:
        columns = get_column_names(table)
        all_columns[table] = columns
        
        # Identificar posibles columnas de fecha/tiempo
        for col in columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['time', 'date', 'timestamp', 'created', 'updated', 
                                                'fecha', 'hora', 'tiempo']):
                date_columns.append((table, col))
    
    # Si no hay columnas de fecha, devolver la consulta original
    if not date_columns:
        return query
    
    # Verificar si la consulta es un SELECT simple
    query_lower = query.lower().strip()
    if not query_lower.startswith('select '):
        return query
    
    # Para consultas SELECT, modificar para formatear fechas
    modified_query = query
    
    # Buscar SELECT y FROM en la consulta
    try:
        select_pos = query_lower.find('select ')
        from_pos = query_lower.find(' from ')
        
        if select_pos != -1 and from_pos != -1:
            select_clause = query[select_pos + 7:from_pos].strip()
            
            # Si es SELECT *, expandir a todas las columnas y aplicar formato a las de fecha
            if select_clause == '*':
                # Intentar determinar la tabla de la cláusula FROM
                rest_of_query = query_lower[from_pos + 6:]
                table_clause_end = min(pos for pos in [
                    rest_of_query.find(' where '), 
                    rest_of_query.find(' group by '),
                    rest_of_query.find(' order by '),
                    rest_of_query.find(' limit '),
                    len(rest_of_query)
                ] if pos != -1)
                
                table_name = rest_of_query[:table_clause_end].strip().split(' ')[0].strip()
                
                if table_name in all_columns:
                    new_select_parts = []
                    
                    for col in all_columns[table_name]:
                        if any((table_name, col) == date_col for date_col in date_columns):
                            new_select_parts.append(f"datetime({col}, 'unixepoch', 'localtime') AS {col}")
                        else:
                            new_select_parts.append(col)
                    
                    new_select_clause = ", ".join(new_select_parts)
                    modified_query = query.replace("*", new_select_clause)
            
            # Si no es SELECT *, buscar columnas de fecha en la cláusula SELECT
            else:
                for table, col in date_columns:
                    # Buscar si la columna está en el SELECT
                    if col in select_clause or f"{table}.{col}" in select_clause:
                        # Reemplazar la columna con su versión formateada
                        if f"{table}.{col}" in select_clause:
                            old_col = f"{table}.{col}"
                            new_col = f"datetime({table}.{col}, 'unixepoch', 'localtime') AS {col}"
                        else:
                            old_col = col
                            new_col = f"datetime({col}, 'unixepoch', 'localtime') AS {col}"
                        
                        # Reemplazar solo si está como columna independiente
                        parts = select_clause.split(',')
                        for i, part in enumerate(parts):
                            part = part.strip()
                            if part == old_col or part.endswith(f" AS {col}") or part.endswith(f" as {col}"):
                                parts[i] = new_col
                        
                        new_select_clause = ", ".join(parts)
                        modified_query = query.replace(select_clause, new_select_clause)
    except:
        # Si hay algún error, devolver la consulta original
        return query
    
    return modified_query

def ask_database(query):
    """Ejecuta una consulta SQL y devuelve los resultados con formato de fecha legible."""
    try:
        # Validar que la consulta es segura
        if any(keyword in query.lower() for keyword in ['drop', 'delete', 'truncate', 'alter']):
            return {
                "query": query,
                "success": False,
                "error": "Error: La consulta contiene palabras clave potencialmente peligrosas."
            }
            
        # Obtener conexión a la base de datos
        db = get_db()
        
        # Convertir resultados a formato lista de diccionarios para mejor procesamiento
        cursor = db.execute(query)
        column_names = [description[0] for description in cursor.description]
        results = cursor.fetchall()
        
        # Si los resultados son muy largos, limitar la salida
        MAX_ROWS = 100
        truncated = False
        if len(results) > MAX_ROWS:
            results = results[:MAX_ROWS]
            truncated = True
            
        # Buscar columnas de timestamp o fecha por nombre común
        time_column_patterns = [
            'time', 'date', 'timestamp', 'created', 'modified', 'updated',
            'fecha', 'hora', 'tiempo', 'creado', 'modificado', 'actualizado'
        ]
        
        # Identificar posibles columnas de timestamp
        time_columns_indices = []
        for i, col_name in enumerate(column_names):
            col_lower = col_name.lower()
            if any(pattern in col_lower for pattern in time_column_patterns):
                time_columns_indices.append(i)
            
        # Convertir a lista de diccionarios y procesar fechas
        formatted_results = []
        for row in results:
            row_list = list(row)
            
            # Intentar convertir posibles timestamps UNIX a formato legible
            for idx in time_columns_indices:
                try:
                    value = row_list[idx]
                    # Verificar si es un número (posible timestamp UNIX)
                    if isinstance(value, (int, float)) and value > 1000000000:  # Probablemente timestamp UNIX
                        # Convertir usando datetime de Python
                        dt = datetime.fromtimestamp(value)
                        row_list[idx] = dt.strftime('%d/%m/%y %H:%M:%S')
                except:
                    # Si falla la conversión, dejar el valor original
                    pass
                    
            formatted_results.append(dict(zip(column_names, row_list)))
            
        # Preparar respuesta
        response = {
            "query": query,
            "success": True,
            "results_count": len(results),
            "truncated": truncated,
            "column_names": column_names,
            "results": formatted_results
        }
        
        return response
        
    except Exception as e:
        error_info = {
            "query": query,
            "success": False,
            "error": str(e)
        }
        return error_info

def create_visualization(results, visualization_type, title="", x_label="", y_label="", 
                         categorical=False, chart_style="darkgrid", color_palette="viridis"):
    """
    Crea una visualización basada en los resultados de una consulta SQL.
    
    Args:
        results: Resultados de la consulta SQL (formato dict o lista)
        visualization_type: Tipo de visualización ('bar', 'line', 'pie', 'scatter', 'heatmap', etc.)
        title: Título del gráfico
        x_label: Etiqueta para el eje X
        y_label: Etiqueta para el eje Y
        categorical: Indica si los datos son categóricos
        chart_style: Estilo del gráfico ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
        color_palette: Paleta de colores para el gráfico
        
    Returns:
        Imagen codificada en base64
    """
    print(f"\n🔍 DEBUG - Creando visualización de tipo: {visualization_type}")
    
    # Verificar y procesar los resultados
    if not results:
        return None, "No hay datos para visualizar"
    
    # Si results es un diccionario con la estructura de respuesta de ask_database
    if isinstance(results, dict) and 'results' in results:
        results_data = results['results']
    else:
        results_data = results
    
    # Convertir resultados a DataFrame
    try:
        if isinstance(results_data, list):
            if isinstance(results_data[0], dict):
                # Lista de diccionarios
                df = pd.DataFrame(results_data)
            elif isinstance(results_data[0], (list, tuple)):
                # Lista de listas/tuplas
                if len(results_data[0]) == 2:
                    df = pd.DataFrame(results_data, columns=['x', 'y'])
                else:
                    df = pd.DataFrame(results_data)
                    df.columns = ['x', 'y'] + [f'col_{i}' for i in range(2, df.shape[1])]
            else:
                return None, f"Formato de datos no compatible: {type(results_data[0])}"
        else:
            return None, f"Tipo de resultados no compatible: {type(results_data)}"
            
        # Seleccionar columnas para visualización si el DataFrame tiene múltiples columnas
        if df.shape[1] > 2 and 'x' not in df.columns and 'y' not in df.columns:
            # Intentar identificar columnas numéricas para y y no numéricas para x
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            
            if numeric_cols and non_numeric_cols:
                # Preferir columnas con nombres relacionados con tiempo/fecha para x
                time_related_cols = [col for col in non_numeric_cols if any(term in col.lower() 
                                    for term in ['fecha', 'date', 'time', 'año', 'mes', 'dia'])]
                
                x_col = time_related_cols[0] if time_related_cols else non_numeric_cols[0]
                y_col = numeric_cols[0]
            elif numeric_cols:
                if len(numeric_cols) >= 2:
                    x_col = numeric_cols[0]
                    y_col = numeric_cols[1]
                else:
                    # Si solo hay una columna numérica, usarla para y y crear un índice para x
                    df['_index'] = range(len(df))
                    x_col = '_index'
                    y_col = numeric_cols[0]
            elif non_numeric_cols:
                if len(non_numeric_cols) >= 2:
                    x_col = non_numeric_cols[0]
                    # Contar frecuencias para y
                    df = df.groupby(non_numeric_cols[0]).size().reset_index(name='count')
                    x_col = non_numeric_cols[0]
                    y_col = 'count'
                else:
                    return None, "No hay suficientes columnas utilizables para visualización"
            else:
                return None, "No se pueden determinar columnas adecuadas para visualización"
                
            # Renombrar columnas para facilitar el procesamiento
            df = df.rename(columns={x_col: 'x', y_col: 'y'})
    except Exception as e:
        return None, f"Error al procesar datos: {str(e)}"
    
    # Limpiar cualquier gráfico anterior
    plt.close('all')
        
    # Configurar estilo
    sns.set_style(chart_style)
    plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Crear visualización según el tipo solicitado
    try:
        if visualization_type == 'bar':
            if categorical:
                sns.barplot(x=df['x'], y=df['y'], palette=color_palette, ax=ax)
            else:
                df.plot(kind='bar', x='x', y='y', color=sns.color_palette(color_palette, len(df)), ax=ax)
                
        elif visualization_type == 'line':
            df.plot(kind='line', x='x', y='y', ax=ax, marker='o')
            
        elif visualization_type == 'pie':
            df.plot(kind='pie', y='y', labels=df['x'], autopct='%1.1f%%', 
                   colors=sns.color_palette(color_palette, len(df)), ax=ax)
            plt.axis('equal')
            
        elif visualization_type == 'scatter':
            if df.shape[1] > 2:  # Si hay una tercera columna, usarla para el tamaño
                sizes = df.iloc[:, 2] * 20
                sizes = sizes.clip(lower=20)  # Asegurar tamaño mínimo
                plt.scatter(df['x'], df['y'], s=sizes, 
                           alpha=0.7, c=range(len(df)), cmap=color_palette)
            else:
                plt.scatter(df['x'], df['y'], s=100, alpha=0.7, 
                           c=range(len(df)), cmap=color_palette)
                
        elif visualization_type == 'heatmap':
            # Para heatmap procesamos los datos de manera especial
            try:
                # Intentamos crear una tabla pivote
                if df.shape[1] > 2:
                    pivot_table = df.pivot(index='x', columns='y', values=df.columns[2])
                else:
                    # Si solo tenemos x,y intentamos agrupar y contar
                    counts = df.groupby(['x', 'y']).size().unstack(fill_value=0)
                    pivot_table = counts
                
                sns.heatmap(pivot_table, annot=True, cmap=color_palette, ax=ax)
            except:
                # Fallback: crear una matriz de correlación con los datos numéricos
                numeric_df = df.select_dtypes(include=['number'])
                if len(numeric_df.columns) > 1:
                    corr = numeric_df.corr()
                    sns.heatmap(corr, annot=True, cmap=color_palette, ax=ax)
                else:
                    return None, "No se pueden crear heatmaps con los datos proporcionados"
            
        elif visualization_type == 'histogram':
            if pd.api.types.is_numeric_dtype(df['y']):
                sns.histplot(data=df, x='y', kde=True, 
                            color=sns.color_palette(color_palette)[0], ax=ax)
            else:
                sns.histplot(data=df, x='x', kde=True,
                            color=sns.color_palette(color_palette)[0], ax=ax)
        
        else:
            return None, f"Tipo de visualización '{visualization_type}' no soportado"
        
        # Añadir etiquetas de datos en gráficos de barras
        if visualization_type == 'bar':
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.1f}',
                           (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='center',
                           xytext=(0, 9),
                           textcoords='offset points')
            
        # Configurar etiquetas y título
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        
        if visualization_type != 'pie':
            plt.xticks(rotation=45 if len(df) > 5 else 0)
            
        # Ajustar diseño y límites
        plt.tight_layout()
        
        # Guardar el gráfico como backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        direct_filename = f"direct_visualization_{timestamp}.png"
        plt.savefig(direct_filename, format='png', dpi=150, bbox_inches='tight')
        
        # Convertir el gráfico a imagen en base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close('all')
        
        # Codificar en base64
        graphic = base64.b64encode(image_png).decode('utf-8')
        return graphic, None
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        plt.close('all')
        return None, f"Error al crear el gráfico: {str(e)}"

def display_graphic(graphic):
    """Muestra un gráfico codificado en base64 y lo abre automáticamente."""
    try:
        # Esto funciona en entornos como Jupyter Notebook
        from IPython.display import Image, display
        display(Image(data=base64.b64decode(graphic)))
        print("\n✅ Visualización mostrada en notebook.")
        return
    except ImportError:
        # Para entorno de terminal, guardamos la imagen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visualization_{timestamp}.png"
        absolute_path = os.path.abspath(filename)
        
        with open(filename, "wb") as f:
            f.write(base64.b64decode(graphic))
        
        print(f"\n✅ VISUALIZACIÓN GUARDADA EN: {absolute_path}")
        print(f"   Intentando abrir automáticamente...")
        
        # Determinar el sistema operativo y abrir la imagen automáticamente
        try:
            # Función para verificar si un comando existe
            def command_exists(cmd):
                if os.name == 'nt':  # Windows
                    from subprocess import DEVNULL, call
                    return call(['where', cmd], stdout=DEVNULL, stderr=DEVNULL) == 0
                else:  # Unix/Linux/Mac
                    from shutil import which
                    return which(cmd) is not None
            
            opened = False
            
            # Intento 1: Método específico del sistema operativo
            if os.name == 'nt':  # Windows
                try:
                    os.startfile(filename)
                    print("   ✓ Imagen abierta automáticamente con visualizador predeterminado.")
                    opened = True
                except Exception as e:
                    print(f"   ⚠️ No se pudo abrir con método de Windows: {e}")
            
            elif sys.platform == 'darwin':  # macOS
                if command_exists('open'):
                    result = os.system(f'open "{filename}"')
                    if result == 0:
                        print("   ✓ Imagen abierta automáticamente con 'open'.")
                        opened = True
                    else:
                        print(f"   ⚠️ El comando 'open' falló con código: {result}")
                else:
                    print("   ⚠️ Comando 'open' no disponible en macOS.")
            
            else:  # Linux/Unix
                viewers = ['xdg-open', 'display', 'eog', 'feh', 'gwenview', 'gthumb', 'gimp']
                for viewer in viewers:
                    if command_exists(viewer):
                        result = os.system(f'{viewer} "{filename}" &')
                        if result == 0:
                            print(f"   ✓ Imagen abierta automáticamente con {viewer}.")
                            opened = True
                            break
                        else:
                            print(f"   ⚠️ El comando {viewer} falló con código: {result}")
            
            # Intento 2: Método alternativo usando Python
            if not opened:
                try:
                    import webbrowser
                    file_url = f"file://{os.path.abspath(filename)}"
                    if webbrowser.open(file_url):
                        print("   ✓ Imagen abierta automáticamente con navegador web.")
                        opened = True
                    else:
                        print("   ⚠️ No se pudo abrir con navegador web.")
                except Exception as e:
                    print(f"   ⚠️ Error al intentar abrir con webbrowser: {e}")
            
            # Si nada funcionó, mostrar instrucciones claras
            if not opened:
                print("\n⚠️ NO SE PUDO ABRIR LA IMAGEN AUTOMÁTICAMENTE.")
                print("\nPor favor, abre manualmente el archivo:")
                print(f"   📂 {absolute_path}")
                
        except Exception as e:
            print(f"   ⚠️ Error al intentar abrir la imagen: {e}")
            print(f"\n📂 La imagen está guardada en: {absolute_path}")

def save_conversation(conversation_history):
    """Guarda la conversación actual en un archivo de texto."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        for entry in conversation_history:
            role = entry["role"]
            content = entry["content"]
            f.write(f"[{role.upper()}]:\n{content}\n\n")
    
    return filename

from dotenv import load_dotenv
import os

def get_api_key():
    """Obtiene la API key desde el archivo .env, sobrescribiendo cualquier valor existente"""
    try:
        # Forzar la recarga para sobrescribir cualquier variable existente
        load_dotenv(override=True)
        
        # Intentar cargar la clave de la variable de entorno después de recargar
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Si aún no funciona, intentar leer directamente del archivo .env
        if not api_key:
            env_path = os.path.join(os.path.dirname(__file__), '.env')
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY='):
                            api_key = line.strip().split('=', 1)[1]
                            # Eliminar comillas si existen
                            api_key = api_key.strip('"\'')
                            
                            # Establecer explícitamente la variable de entorno
                            os.environ["OPENAI_API_KEY"] = api_key
                            break
        
        if not api_key:
            raise ValueError("No se encontró la API key de OpenAI. Por favor, configúrala en el archivo .env")
        
        # Imprimir los primeros y últimos caracteres para debug (sin exponer toda la clave)
        if api_key:
            masked_key = f"{api_key[:4]}...{api_key[-4:]}"
            print(f"API key cargada: {masked_key}")
        
        return api_key
    except Exception as e:
        print(f"Error al cargar la API key: {str(e)}")
        raise

def process_documents(api_key):
    """Procesa los documentos disponibles y crea/actualiza la base de datos vectorial."""
    print("\n📚 Procesando documentos...")
    
    # Verificar si hay documentos
    document_files = []
    for ext in ['*.pdf', '*.txt', '*.docx', '*.doc', '*.csv', '*.xlsx', '*.xls']:
        document_files.extend(glob.glob(os.path.join(DOCUMENT_DIR, ext)))
    
    if not document_files:
        print("⚠️ No se encontraron documentos")
        if not document_files:
            return None, []
    
    documents = []
    processed_files = []
    
    # Cargar cada documento
    for file_path in document_files:
        try:
            print(f"  📄 Procesando: {os.path.basename(file_path)}")
            loader = get_file_loader(file_path)
            file_documents = loader.load()
            documents.extend(file_documents)
            processed_files.append(os.path.basename(file_path))
        except Exception as e:
            print(f"  ❌ Error al procesar {file_path}: {str(e)}")
    
    if not documents:
        return None, []
    
    # Dividir documentos en chunks para mejor procesamiento
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    split_documents = text_splitter.split_documents(documents)
    print(f"  🔄 Total de chunks generados: {len(split_documents)}")
    
    # Crear embeddings y base de datos vectorial
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Crear o actualizar la base de datos vectorial
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        # Cargar la base existente y actualizar
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        # Añadir nuevos documentos
        db.add_documents(split_documents)
    else:
        # Crear nueva base de datos
        db = Chroma.from_documents(
            documents=split_documents,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
    
    # Persistir la base de datos
    db.persist()
    print(f"✅ Base de datos vectorial actualizada con {len(split_documents)} fragmentos.")
    
    return db, processed_files

    # Funciones para el manejo de documentos

def setup_document_directory():
    """Crea el directorio de documentos si no existe."""
    os.makedirs(DOCUMENT_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    print(f"✅ Directorios configurados: {DOCUMENT_DIR} y {CHROMA_DIR}")

def get_file_loader(file_path: str):
    """Retorna el cargador adecuado según la extensión del archivo."""
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.pdf':
        return PyPDFLoader(file_path)
    elif file_extension in ['.docx', '.doc']:
        return Docx2txtLoader(file_path)
    elif file_extension == '.txt':
        return TextLoader(file_path)
    elif file_extension == '.csv':
        return CSVLoader(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        return UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Tipo de archivo no soportado: {file_extension}")
        
def query_documents(db: Chroma, query: str, n_results: int = 5) -> List[Document]:
    """
    Consulta la base de datos vectorial para encontrar documentos relevantes.
    
    Args:
        db: Base de datos vectorial Chroma
        query: Consulta del usuario
        n_results: Número de resultados a retornar
        
    Returns:
        List[Document]: Lista de documentos relevantes
    """
    if db is None:
        return []
    
    results = db.similarity_search(query, k=n_results)
    return results

def extract_document_context(results: List[Document]) -> str:
    """
    Extrae el contexto de los documentos encontrados para enriquecer la consulta.
    
    Args:
        results: Lista de documentos relevantes
        
    Returns:
        str: Contexto extraído de los documentos
    """
    if not results:
        return ""
    
    context = "\n\n".join([f"--- Fragmento de documento ---\n{doc.page_content}" for doc in results])
    return context

def upload_document(file_path: str) -> bool:
    """
    Copia un documento al directorio de documentos.
    
    Args:
        file_path: Ruta del archivo a copiar
        
    Returns:
        bool: True si se copió correctamente, False en caso contrario
    """
    try:
        filename = os.path.basename(file_path)
        destination = os.path.join(DOCUMENT_DIR, filename)
        
        # Copiar el archivo
        import shutil
        shutil.copy2(file_path, destination)
        
        print(f"✅ Documento '{filename}' añadido correctamente.")
        return True
    except Exception as e:
        print(f"❌ Error al añadir documento: {str(e)}")
        return False

def get_document_status() -> Dict[str, Any]:
    """
    Obtiene el estado actual de los documentos.
    
    Returns:
        Dict: Información sobre los documentos disponibles
    """
    document_files = []
    for ext in ['*.pdf', '*.txt', '*.docx', '*.doc', '*.csv', '*.xlsx', '*.xls']:
        document_files.extend(glob.glob(os.path.join(DOCUMENT_DIR, ext)))
    
    # Agrupar por tipo
    doc_types = {}
    for doc in document_files:
        ext = Path(doc).suffix.lower()
        if ext not in doc_types:
            doc_types[ext] = 0
        doc_types[ext] += 1
    
    return {
        "total_documents": len(document_files),
        "documents_by_type": doc_types,
        "document_names": [os.path.basename(f) for f in document_files]
    }

def main():
    # Configuración inicial
    setup_document_directory()
    
    # Configuración de OpenAI
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)
    
    # Procesar documentos y crear/actualizar base de datos vectorial
    doc_db, processed_files = process_documents(api_key)
    
    # Obtener información detallada de la base de datos
    detailed_schema = generate_database_schema_string()
    
    # Estado de los documentos
    doc_status = get_document_status()
    documents_info = f"""
    Total de documentos: {doc_status['total_documents']}
    Tipos de documentos: {', '.join([f"{k}: {v}" for k, v in doc_status['documents_by_type'].items()])}
    Documentos disponibles: {', '.join(doc_status['document_names'])}
    """
    
    # Configuramos las herramientas para OpenAI
    db_tool = {
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
                            {detailed_schema}
                            The query should be specific and directly address the user's question.
                            Always use SELECT statements only. Never use DROP, DELETE, UPDATE, or ALTER statements.
                            Limit your results to avoid excessive data.
                        """
                    }
                },
                "required": ["query"]
            }
        }
    }
    
    visualization_tool = {
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
    }
    
    # Herramienta para consultar documentos
    document_tool = {
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
    }
    
    # Sistema de gestión de conversación
    conversation_history = [
        {
            "role": "system",
            "content": f"""Eres un asistente virtual avanzado especializado en consultas de bases de datos y análisis de datos.

CAPACIDADES PRINCIPALES:

- Consultas SQL avanzadas con optimización de rendimiento
- Visualización de datos dinámica y adaptativa
- Análisis semántico de documentos y correlación con datos estructurados
- Análisis multidimensional y estadísticas predictivas
- Generación y optimización de código para manipulación de datos

DIRECTRICES PARA CONSULTAS A LA BASE DE DATOS:
- Utiliza técnicas de procesamiento de lenguaje natural para interpretar la intención del usuario
- Implementa consultas SQL optimizadas, considerando índices y planes de ejecución eficientes
- Aplica validación cruzada de resultados utilizando múltiples métodos de consulta
- En caso de ambigüedad, genera preguntas de seguimiento específicas y contextuales
- Realiza un análisis exhaustivo de esquemas y relaciones entre tablas para consultas completas
- Para cálculos numéricos, implementa controles de calidad y verificación de outliers
- Proporciona explicaciones detalladas de los resultados, incluyendo contexto histórico y tendencias
- Sugiere visualizaciones y análisis adicionales basados en patrones identificados en los datos

DIRECTRICES PARA VISUALIZACIONES:
- Utiliza algoritmos de selección automática para recomendar el tipo de gráfico óptimo
- Implementa títulos dinámicos y etiquetas que resalten insights clave de los datos
- Ajusta automáticamente escalas, formatos y paletas de colores para maximizar la claridad
- Genera descripciones narrativas que expliquen las tendencias y anomalías en la visualización
- Incorpora elementos interactivos para exploración profunda de los datos
- Contextualiza las visualizaciones con benchmarks relevantes y datos históricos

CAPACIDADES ANALÍTICAS AVANZADAS:
- Implementa análisis de series temporales para identificar patrones y tendencias
- Utiliza técnicas de machine learning para clasificación y predicción de datos
- Realiza análisis de correlación entre múltiples variables y tablas
- Implementa detección automática de anomalías y valores atípicos
- Genera informes automáticos con resúmenes ejecutivos de los hallazgos clave
- Ten en cuenta que el tiempo de rodaje es el tiempo desde que se enciende el ADS-B y tenemos señal hasta que al altura es <100ft
INTEGRACIÓN Y CONTEXTUALIZACIÓN:
- Correlaciona datos estructurados con información de documentos no estructurados
- Incorpora fuentes de datos externas relevantes para enriquecer el análisis
- Mantén un registro de consultas frecuentes y resultados para aprendizaje continuo
- Implementa un sistema de retroalimentación para mejorar la precisión de las respuestas

INFORMACIÓN DE LA BASE DE DATOS:
{detailed_schema}

INFORMACIÓN DE DOCUMENTOS DISPONIBLES:
{documents_info}

Cuando no estés seguro de la respuesta o los datos solicitados no estén disponibles, indícalo claramente y ofrece alternativas.
"""
        }
    ]

    print("=" * 60)
    print("ASISTENTE AVANZADO DE CONSULTAS Y ANÁLISIS DE DATOS")
    print("=" * 60)
    print("Capacidades principales:")
    print("• Consultas precisas a la base de datos")
    print("• Visualización interactiva de datos")
    print("• Búsqueda contextual en documentos")
    print("• Análisis estadístico e interpretación")
    print("• Asistencia con SQL y programación de datos")
    print("\nComandos disponibles:")
    print("- 'historia': Muestra el historial de la conversación")
    print("- 'guardar': Guarda la conversación actual en un archivo")
    print("- 'documentos': Muestra información sobre documentos disponibles")
    print("- 'subir [ruta]': Sube un nuevo documento al sistema")
    print("- 'tablas': Muestra las tablas disponibles en la base de datos")
    print("- 'salir': Termina el programa")
    print("=" * 60)
    
    while True:
        # Solicitar la pregunta al usuario
        user_question = input("\n👤 Tú: ").strip()
        
        # Verificar comandos especiales
        if user_question.lower() == 'salir':
            print("¡Hasta luego! Gracias por usar el asistente.")
            break
            
        elif user_question.lower() == 'historia':
            print("\n📜 HISTORIAL DE CONVERSACIÓN:")
            for entry in conversation_history:
                if entry["role"] != "system":
                    content_preview = entry['content'][:150] + '...' if len(entry['content']) > 150 else entry['content']
                    print(f"[{entry['role'].upper()}]: {content_preview}")
            continue
            
        elif user_question.lower() == 'guardar':
            filename = save_conversation(conversation_history)
            print(f"\n✅ Conversación guardada en: {filename}")
            continue
            
        elif user_question.lower() == 'documentos':
            status = get_document_status()
            print("\n📚 DOCUMENTOS DISPONIBLES:")
            print(f"Total: {status['total_documents']} documentos")
            if status['documents_by_type']:
                print("Tipos:")
                for ext, count in status['documents_by_type'].items():
                    print(f"  - {ext}: {count} archivos")
                print("\nLista de documentos:")
                for doc in status['document_names']:
                    print(f"  - {doc}")
            else:
                print("No hay documentos disponibles en el sistema.")
            continue
            
        elif user_question.lower() == 'tablas':
            tables = get_table_names()
            print("\n📊 TABLAS DISPONIBLES EN LA BASE DE DATOS:")
            for i, table in enumerate(tables, 1):
                columns = get_column_names(table)
                print(f"{i}. {table} ({len(columns)} columnas)")
                # Mostrar primeras 5 columnas como ejemplo
                preview = ", ".join(columns[:5])
                if len(columns) > 5:
                    preview += ", ..."
                print(f"   Columnas: {preview}")
            continue
            
        elif user_question.lower().startswith('subir '):
            file_path = user_question[6:].strip()
            if os.path.exists(file_path):
                success = upload_document(file_path)
                if success:
                    # Actualizar la base de datos vectorial
                    doc_db, _ = process_documents(api_key)
                    # Actualizar información de documentos
                    doc_status = get_document_status()
                    documents_info = f"""
                    Total de documentos: {doc_status['total_documents']}
                    Tipos de documentos: {', '.join([f"{k}: {v}" for k, v in doc_status['documents_by_type'].items()])}
                    Documentos disponibles: {', '.join(doc_status['document_names'])}
                    """
                    # Actualizar el contexto del sistema
                    system_content = conversation_history[0]["content"]
                    updated_content = system_content.replace(
                        "INFORMACIÓN DE DOCUMENTOS DISPONIBLES:", 
                        f"INFORMACIÓN DE DOCUMENTOS DISPONIBLES:\n{documents_info}"
                    )
                    conversation_history[0]["content"] = updated_content
            else:
                print(f"\n❌ Error: El archivo '{file_path}' no existe.")
            continue
        
        # Añadir la pregunta del usuario al historial
        conversation_history.append({"role": "user", "content": user_question})
        
        # Analizar la intención de la consulta
        query_intent = analyze_query_intent(user_question)
        
        # Determinar qué herramientas usar basado en la intención
        needs_db_query = query_intent["needs_db_query"]
        needs_date_formatting = query_intent.get("needs_date_formatting", False)
        
        needs_visualization = any(keyword in user_question.lower() for keyword in 
                                 ['gráfico', 'visualiza', 'grafica', 'mostrar', 'ver', 'plot', 'chart',
                                  'diagrama', 'barras', 'líneas', 'pastel', 'dispersión', 'histograma'])
        
        needs_document_query = any(keyword in user_question.lower() for keyword in 
                                  ['documento', 'pdf', 'archivo', 'texto', 'buscar en', 'encontrar en',
                                   'información sobre', 'qué dice', 'menciona', 'contiene'])
        
        # Preparar herramientas según necesidades
        tools = []
        if needs_db_query:
            tools.append(db_tool)
        if needs_visualization:
            tools.append(visualization_tool)
        if needs_document_query and doc_db is not None:
            tools.append(document_tool)
        
        tools = tools if tools else None
        
        # Crear la solicitud a OpenAI
        try:
            chat_completion = client.chat.completions.create(
                messages=conversation_history,
                model="gpt-3.5-turbo",
                tools=tools,
                temperature=0.2  # Reducir temperatura para consultas más precisas
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
                
                # Procesar cada llamada a herramienta
                for tool_call in assistant_message.tool_calls:
                    try:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        if function_name == 'ask_database':
                            query = function_args["query"]
                            
                            # Verificar si se necesita formateo de fecha
                            if needs_date_formatting:
                                original_query = query
                                tables = get_table_names()
                                modified_query = modify_query_for_date_formatting(query, tables)
                                if modified_query != original_query:
                                    print(f"\n🔄 Consulta modificada para formato de fecha legible")
                                    query = modified_query
                            
                            print(f"\n🔍 Ejecutando consulta SQL: {query}")
                            function_response = ask_database(query)
                            
                        elif function_name == 'create_visualization':
                            print(f"\n📊 Creando visualización de tipo: {function_args['visualization_type']}")
                            graphic, error = create_visualization(**function_args)
                            if graphic:
                                display_graphic(graphic)
                                function_response = {"success": True, "message": "Visualización creada correctamente"}
                            else:
                                function_response = {"success": False, "error": error}
                                
                        elif function_name == 'query_documents':
                            query = function_args["query"]
                            n_results = function_args.get("n_results", 5)
                            print(f"\n🔎 Buscando en documentos: {query}")
                            
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
                        print(f"\n❌ {error_message}")
                        conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": json.dumps({"error": error_message})
                        })
                
                # Obtener la respuesta final del asistente
                final_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=conversation_history,
                    temperature=0.2
                )
                
                assistant_response = final_response.choices[0].message.content
                
            else:
                # Si no hay llamadas a herramientas, usamos la respuesta directamente
                assistant_response = assistant_message.content
            
            # Imprimir la respuesta del asistente
            print("\n🤖 Asistente:")
            wrapped_text = textwrap.fill(assistant_response, width=100)
            print(wrapped_text)
            
            # Añadir la respuesta final al historial
            conversation_history.append({"role": "assistant", "content": assistant_response})
            
        except Exception as e:
            error_message = f"Error en la comunicación con OpenAI: {str(e)}"
            print(f"\n❌ {error_message}")
            conversation_history.append({"role": "assistant", "content": f"Lo siento, ocurrió un error: {error_message}"})

# Ejecutar el programa principal cuando se llama directamente
if __name__ == "__main__":
    main()