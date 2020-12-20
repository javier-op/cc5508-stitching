# cc5508-stitching

El programa stitch.py es una implementación de stitching de imagenes que utiliza
SIFT para obtener descriptores locales y estima una transformación geométrica,
se necesita Python 3 instalado con las librerías descritas en requirements.txt.

Se ejecuta con la siguiente linea:

python stitch.py [path to first image] [path to second image]

Se espera que la primera imagen sea la que va a la izquierda, de forma que a la
segunda se le realiza la operación warp y se pega a la derecha de la primera.

El resultado son dos imágenes: una con los matches entre descriptores locales
y la otra es el resultado de la operación de stitching. Ambas imágenes se
guardan en el directorio en que se encuentra el programa.
