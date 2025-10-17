import os
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from movie.models import Movie
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

class Command(BaseCommand):
    help = "Calcula similitudes entre películas usando OpenAI embeddings"

    def add_arguments(self, parser):
        parser.add_argument("--topk", type=int, default=5, help="Número de películas similares a guardar")

    def handle(self, *args, **options):
        # ✅ Cargar variables de entorno
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        movies = list(Movie.objects.all())
        texts = [f"{m.title}. {getattr(m, 'overview', '')}" for m in movies]

        self.stdout.write(f"Generando embeddings para {len(texts)} películas...")

        embeddings = self.get_embeddings(client, texts)

        # ✅ Calcular similitud coseno
        sims = cosine_similarity(embeddings)

        rows = []
        topk = options["topk"]

        for i, movie in enumerate(movies):
            scores = sims[i]
            idxs = np.argsort(-scores)  # orden descendente
            idxs = [j for j in idxs if j != i][:topk]  # excluir self

            similar_ids = [movies[j].id for j in idxs]
            similar_scores = [float(scores[j]) for j in idxs]

            rows.append({
                "movie_id": movie.id,
                "similar_ids": ";".join(map(str, similar_ids)),
                "scores": ";".join(f"{s:.4f}" for s in similar_scores)
            })

            # 👉 Opcional: guardar en la BD si tienes un campo JSON o ManyToMany
            # movie.similar_json = json.dumps(list(zip(similar_ids, similar_scores)))
            # movie.save()

        # ✅ Guardar CSV con resultados
        df = pd.DataFrame(rows)
        df.to_csv("movie_similarities.csv", index=False)

        self.stdout.write(self.style.SUCCESS(f"Similitudes calculadas y guardadas en movie_similarities.csv"))

    def get_embeddings(self, client, texts):
        """
        Obtiene embeddings desde OpenAI (modelo text-embedding-3-small).
        Retorna un np.ndarray (N, D).
        """
        embeddings = []
        for text in texts:
            response = client.embeddings.create(
                model="text-embedding-3-small",  # puedes usar "text-embedding-3-large"
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings)
