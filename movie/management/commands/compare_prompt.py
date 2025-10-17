import os
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

class Command(BaseCommand):
    help = "Compara un prompt con las películas y muestra similitudes"

    def add_arguments(self, parser):
        parser.add_argument("--prompt", type=str, required=True, help="Texto para comparar con películas")
        parser.add_argument("--topk", type=int, default=5, help="Número de películas similares a mostrar")

    def handle(self, *args, **options):
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        prompt_text = options["prompt"]
        topk = options["topk"]

        # ✅ Obtener películas
        movies = list(Movie.objects.all())
        texts = [f"{m.title}. {getattr(m, 'overview', '')}" for m in movies]

        # ✅ Obtener embeddings de películas
        self.stdout.write("Generando embeddings de películas...")
        movie_embeddings = []
        for text in texts:
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            movie_embeddings.append(emb.data[0].embedding)
        movie_embeddings = np.array(movie_embeddings)

        # ✅ Embedding del prompt
        self.stdout.write("Generando embedding del prompt...")
        prompt_emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=prompt_text
        ).data[0].embedding
        prompt_emb = np.array(prompt_emb).reshape(1, -1)

        # ✅ Calcular similitud coseno
        sims = cosine_similarity(prompt_emb, movie_embeddings)[0]

        # ✅ Ordenar y mostrar top K
        idxs = np.argsort(-sims)[:topk]
        self.stdout.write("\nResultados de similitud:\n")
        for idx in idxs:
            self.stdout.write(f"📝 Similitud prompt vs {movies[idx].title}: {sims[idx]:.2f}")

        # ✅ Ejemplo extra: comparar dos películas entre sí
        if len(movies) >= 2:
            sim_movies = cosine_similarity(
                [movie_embeddings[0]], [movie_embeddings[1]]
            )[0][0]
            self.stdout.write(f"\n🎬 {movies[0].title} vs {movies[1].title}: {sim_movies:.2f}")
