# check_openai_key.py

from openai import OpenAI
from dotenv import load_dotenv
import os

# 1️⃣ Carga las variables del archivo .env
load_dotenv()

# 2️⃣ Obtiene la clave del entorno
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ No se encontró la variable OPENAI_API_KEY en el entorno o en el archivo .env")
    exit()

# 3️⃣ Inicializa el cliente de OpenAI
client = OpenAI(api_key=api_key)

print("🔍 Verificando clave de OpenAI...")

try:
    # 4️⃣ Intenta listar modelos disponibles
    models = client.models.list()

    print("✅ Conexión exitosa. La API key es válida.")
    print("📦 Modelos disponibles (primeros 5):")
    for m in models.data[:5]:
        print("   •", m.id)

except Exception as e:
    # 5️⃣ Muestra el mensaje exacto de error
    print("❌ Error al verificar la clave:")
    print(e)
