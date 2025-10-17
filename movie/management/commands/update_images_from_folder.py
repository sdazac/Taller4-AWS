import os
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Asigna imágenes desde la carpeta media/movie/images/ a las películas en la BD"

    def handle(self, *args, **kwargs):
        # ✅ Ruta absoluta de la carpeta de imágenes
        images_folder = r"C:\Users\ASUS\Desktop\P1\moviereviewsproject\DjangoProjectBase\media\movie\images"

        if not os.path.exists(images_folder):
            self.stderr.write(f"La carpeta {images_folder} no existe.")
            return

        movies = Movie.objects.all()
        self.stdout.write(f"Se encontraron {movies.count()} películas en la base de datos.")

        updated_count = 0
        for movie in movies:
            # 👇 Ajusta el nombre de archivo según tu convención real
            # Ejemplo: m_Titulo.png
            image_filename = f"m_{movie.title}.png"
            image_path_full = os.path.join(images_folder, image_filename)

            if os.path.exists(image_path_full):
                # Guardamos solo la ruta relativa para Django
                relative_path = os.path.join("movie", "images", image_filename)
                movie.image = relative_path
                movie.save()
                updated_count += 1
                self.stdout.write(self.style.SUCCESS(f"Imagen asignada a: {movie.title}"))
            else:
                self.stderr.write(f"No se encontró imagen para: {movie.title}")

        self.stdout.write(self.style.SUCCESS(f"Proceso finalizado. {updated_count} películas actualizadas."))
