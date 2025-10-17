import os
import numpy as np
from django.core.management.base import BaseCommand
from dotenv import load_dotenv
from openai import OpenAI
from movie.models import Movie

class Command(BaseCommand):
    help = "Muestra el embedding generado para una película al azar"

    def handle(self, *args, **kwargs):
        # ✅ Cargar la API Key
        load_dotenv('C:/Users/ASUS/Desktop/P1/moviereviewsproject/DjangoProjectBase/.env')
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

        # ✅ Tomar una película al azar
        movie = Movie.objects.order_by("?").first()
        self.stdout.write(self.style.SUCCESS(f"🎬 Película seleccionada: {movie.title}"))

        # ✅ Generar embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=movie.description
        )
        emb = np.array(response.data[0].embedding, dtype=np.float32)

        # ✅ Mostrar parte del embedding en consola
        self.stdout.write("🧩 Embedding generado (primeros 20 valores):")
        self.stdout.write(str(emb[:20]))
