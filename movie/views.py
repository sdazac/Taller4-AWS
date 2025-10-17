import os
import matplotlib.pyplot as plt
import matplotlib
import io
import urllib, base64
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from django.conf import settings
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .models import Movie

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recommend_from_prompt(request):
    query = request.GET.get("q", "")
    best_movie = None
    max_similarity = -1

    if query:
        # Obtener embedding del prompt
        response = client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        )
        prompt_emb = np.array(response.data[0].embedding, dtype=np.float32)

        # Comparar contra embeddings en BD
        for movie in Movie.objects.all():
            movie_emb = np.frombuffer(movie.emb, dtype=np.float32)
            similarity = cosine_similarity(prompt_emb, movie_emb)

            if similarity > max_similarity:
                max_similarity = similarity
                best_movie = movie

    return render(request, "recommend.html", {
        "query": query,
        "movie": best_movie,
        "similarity": max_similarity if best_movie else None
    })


def statistics_view(request):
    matplotlib.use('Agg')
    years = Movie.objects.values_list('year', flat=True).distinct().order_by('year')  
    movie_counts_by_year = {}  
    
    for year in years:  
        if year:
            movies_in_year = Movie.objects.filter(year=year)
        else:
            movies_in_year = Movie.objects.filter(year__isnull=True)
            year = "None"
        
        count = movies_in_year.count()
        movie_counts_by_year[year] = count

    bar_width = 0.5  
    bar_positions = range(len(movie_counts_by_year))  

    plt.bar(bar_positions, movie_counts_by_year.values(), width=bar_width, align='center')
    plt.title('Movies per year')
    plt.xlabel('Year')
    plt.ylabel('Number of movies')
    plt.xticks(bar_positions, movie_counts_by_year.keys(), rotation=90)
    plt.subplots_adjust(bottom=0.3)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')

    return render(request, 'statistics.html', {'graphic': graphic})

def statistics_genre_view(request):
    matplotlib.use('Agg')

    # Obtener todas las películas
    all_movies = Movie.objects.all()

    # Diccionario para almacenar la cantidad de películas por género
    movie_counts_by_genre = {}

    # Recorremos todas las películas
    for movie in all_movies:
        if movie.genre:  # si tiene género
            # considerar solo el primer género (ej: "Action, Adventure" → "Action")
            first_genre = movie.genre.split(',')[0].strip()
        else:
            first_genre = "None"
        
        if first_genre in movie_counts_by_genre:
            movie_counts_by_genre[first_genre] += 1
        else:
            movie_counts_by_genre[first_genre] = 1

    # Crear la gráfica de barras
    bar_width = 0.5
    bar_positions = range(len(movie_counts_by_genre))

    plt.bar(bar_positions, movie_counts_by_genre.values(), width=bar_width, align='center')
    plt.title('Movies per Genre')
    plt.xlabel('Genre')
    plt.ylabel('Number of movies')
    plt.xticks(bar_positions, movie_counts_by_genre.keys(), rotation=45, ha="right")
    plt.subplots_adjust(bottom=0.3)

    # Guardar la gráfica en un objeto BytesIO
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Convertir la gráfica a base64
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')

    # Renderizar en template
    return render(request, 'statistics_genre.html', {'graphic': graphic})


def signup(request):
    email = request.GET.get('email')
    return render(request, 'signup.html', {'email':email})

def home(request):
    searchTerm = request.GET.get('searchMovie')
    if searchTerm:
        movies = Movie.objects.filter(title__icontains=searchTerm)
    else:
        movies = Movie.objects.all()
    return render(request, 'home.html', {'searchTerm': searchTerm, 'movies': movies})

def about(request):
    return HttpResponse('<h1>Welcome to About Page</h1>')
