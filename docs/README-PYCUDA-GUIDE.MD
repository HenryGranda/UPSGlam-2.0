PyCUDA Service – Procesamiento de Imágenes en GPU ⚙️🧠
1. Descripción general
El PyCUDA Service es un microservicio HTTP, escrito en Python + PyCUDA, que se encarga únicamente de:
●	Recibir una imagen (JPEG/PNG) por HTTP.

●	Aplicar un filtro de convolución en GPU (CUDA).

●	Devolver la imagen filtrada en el mismo formato (bytes de imagen).

No sabe nada de usuarios, posts, likes, Firestore ni Firebase.
 Solo hace: imagen de entrada → GPU → imagen de salida.
________________________________________
2. Endpoints
2.1. POST /filters/{filterName}
Aplica un filtro a una imagen.
●	URL (interno, llamado desde post-service)
 POST http://pycuda-service:5000/filters/{filterName}

●	Path param:

○	filterName (string) – nombre lógico del filtro, por ejemplo:

■	gaussian

■	sharpen

■	edge_detect

■	emboss

■	ups_logo

■	ups_color

●	Headers:

Content-Type: image/jpeg      # o image/png
Accept: image/jpeg            # o image/png, idealmente mismo formato que envías

●	Body (request):

○	Bytes de la imagen (no JSON, no multipart).

○	El cliente (post-service) envía la imagen cruda.

●	Response (200 OK):

Content-Type: image/jpeg
Body: [bytes de la imagen ya filtrada]

●	Errores:

●	400 BAD_REQUEST

○	filterName no existe.

○	Formato de imagen inválido.

●	500 INTERNAL_SERVER_ERROR

○	Error en kernel CUDA, memoria GPU, etc.

Ejemplo conceptual de llamada (desde post-service – cURL):
curl -X POST "http://pycuda-service:5000/filters/gaussian" \
  -H "Content-Type: image/jpeg" \
  --data-binary "@input.jpg" \
  -o "output_gaussian.jpg"

________________________________________
3. Catálogo de filtros
Puedes ajustar los nombres exactos, pero la idea es tener mínimo:
●	4 filtros vistos en clase.

●	2 filtros creativos, uno relacionado con la UPS.

Filtro: gaussian
●	Tipo: convolución blur.

●	Efecto: desenfoque suave (suaviza ruido en la imagen).

●	Kernel típico 3x3 o 5x5 (ejemplo conceptual):

[ 1  2  1
  2  4  2
  1  2  1 ] / 16

Filtro: sharpen
●	Tipo: realce de bordes.

●	Efecto: la imagen se ve más nítida.

●	Kernel típico:

[  0  -1   0
  -1   5  -1
   0  -1   0 ]

Filtro: edge_detect
●	Tipo: detección de bordes (estilo Sobel/Laplace).

●	Efecto: resalta contornos.

Ejemplo: Laplaciano
[ -1  -1  -1
  -1   8  -1
  -1  -1  -1 ]

Filtro: emboss (creativo 1)
●	Tipo: relieve.

●	Efecto: apariencia “en relieve” (como grabado).

Ejemplo kernel:
[ -2  -1   0
  -1   1   1
   0   1   2 ]

Filtro: ups_logo (creativo 2, UPS requerido)
●	Tipo: convolución + overlay.

●	Idea posible:

○	Aplicar un suavizado ligero.

○	Superponer el logo de la UPS en una esquina (o patrón).

●	Implementación simple:

○	Filtro base (ej. gaussian) + escribir el logo en una región fija de la imagen.

○	El logo podría ser una pequeña matriz pre-cargada o aplicar un patrón de color.

Filtro: ups_color (creativo 3, opcional extra)
●	Tipo: colorización.

●	Efecto: aplicar un tinte con los colores de la UPS (ej. dorado/marrón).

●	Implementación:

○	Transformar a escala de grises.

○	Aplicar una función que desplace la paleta de colores hacia el color corporativo.

En el doc del PyCUDA puedes hacer una tablita tipo:
| filterName | Tipo        | Descripción visual                 |
|-----------|-------------|-------------------------------------|
| gaussian  | convolución | Desenfoque suave                   |
| sharpen   | convolución | Imagen más nítida                  |
| edge_detect | convolución | Resalta bordes y contornos      |
| emboss    | convolución | Efecto de relieve                  |
| ups_logo  | creativo    | Blur + logo UPS superpuesto        |
| ups_color | creativo    | Tinte de color con paleta de la UPS|

________________________________________
4. Arquitectura interna del PyCUDA Service
Propuesta de estructura de carpetas:
pycuda-service/
├── Dockerfile
├── requirements.txt
└── src/
    ├── app.py                  # Punto de entrada (Flask/FastAPI)
    ├── api/
    │   └── filters_controller.py  # Define POST /filters/{filterName}
    ├── core/
    │   ├── cuda_context.py     # Inicializa contexto CUDA, device, etc.
    │   ├── kernels/
    │   │   ├── gaussian.cu
    │   │   ├── sharpen.cu
    │   │   ├── edge_detect.cu
    │   │   ├── emboss.cu
    │   │   └── ups_logo.cu     # si lo haces con kernel personalizado
    │   └── filter_service.py   # Lógica para escoger filtro y ejecutar kernel
    └── utils/
        ├── image_io.py         # Bytes ↔ np.ndarray, RGB/GRAY
        └── config.py

4.1. app.py
●	Levanta el servidor HTTP (Flask o FastAPI).

●	Registra la ruta POST /filters/{filterName}.

Pseudocódigo:
from fastapi import FastAPI, Path, Response, HTTPException, Request
from core.filter_service import apply_filter

app = FastAPI()

@app.post("/filters/{filter_name}")
async def apply_filter_endpoint(
    filter_name: str = Path(...),
    request: Request = None
):
    # Leer bytes de la imagen del body
    image_bytes = await request.body()

    try:
        output_bytes = apply_filter(filter_name, image_bytes)
    except UnknownFilterError:
        raise HTTPException(status_code=400, detail="Unknown filter")
    except Exception:
        raise HTTPException(status_code=500, detail="CUDA error")

    return Response(content=output_bytes, media_type="image/jpeg")

4.2. core/filter_service.py
●	Se encarga de:

○	Convertir bytes → array (via image_io.py).

○	Mandar el array al kernel CUDA correcto.

○	Recibir resultado y convertir de vuelta a bytes.

Pseudocódigo:
from utils.image_io import bytes_to_array, array_to_bytes
from core.kernels.gaussian import gaussian_filter_gpu
from core.kernels.sharpen import sharpen_filter_gpu
# etc.

FILTERS = {
    "gaussian": gaussian_filter_gpu,
    "sharpen": sharpen_filter_gpu,
    "edge_detect": edge_detect_filter_gpu,
    "emboss": emboss_filter_gpu,
    "ups_logo": ups_logo_filter_gpu,
    "ups_color": ups_color_filter_gpu,
}

class UnknownFilterError(Exception):
    pass

def apply_filter(filter_name: str, image_bytes: bytes) -> bytes:
    if filter_name not in FILTERS:
        raise UnknownFilterError()

    img = bytes_to_array(image_bytes)          # np.ndarray (H, W, C)
    kernel_fn = FILTERS[filter_name]          # función que ejecuta kernel en GPU
    out_img = kernel_fn(img)                  # np.ndarray filtrado
    out_bytes = array_to_bytes(out_img)       # JPEG bytes

    return out_bytes

4.3. Kernels CUDA
Cada archivo .cu define un kernel CUDA.
Ejemplo conceptual gaussian.cu:
extern "C"
__global__ void gaussian_filter(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels
) {
    // índice global (x, y) y aplicar kernel 3x3 sobre input
}

Y en Python (core/kernels/gaussian.py) usas PyCUDA para:
●	Compilar el kernel.

●	Lanzarlo con los bloques/hilos adecuados.

●	Copiar datos de/to GPU.

________________________________________
5. Flujo de datos con el resto de microservicios
5.1. Estrategia 1 – Procesamiento directo en POST /posts (simple)
1.	App móvil:

○	Hace foto.

○	Llama al backend: POST /posts (multipart) con image + filter.

2.	API Gateway:

○	Redirige a post-service.

3.	post-service:

○	PostHandler.createPostMultipart llama a ImageService.processAndUploadMultipart(...).

○	ImageService:

■	Extrae image y filter.

Llama a PyCUDA:

 POST http://pycuda-service:5000/filters/{filter}
Body: bytes de imagen
■	
■	Recibe los bytes filtrados.

■	Sube la imagen a Storage (posts/{postId}.jpg).

○	Crea documento en posts/{postId} en Firestore.

○	Devuelve el PostResponse a la app.

4.	App móvil:

○	Muestra el post y/o actualiza el feed.

5.2. Estrategia 2 – Preview (/images/preview + tempImageId)
1.	App móvil:

○	Toma foto.

○	Llama a POST /images/preview con image + filter.

2.	post-service → PyCUDA (igual que arriba):

○	PyCUDA aplica el filtro.

○	post-service sube imagen filtrada a Storage temp/{tempImageId}.jpg.

○	Devuelve a la app:

{
  "tempImageId": "tmp-12345",
  "imageUrl": "https://.../temp/tmp-12345.jpg"
}
3.	
4.	App móvil:

○	Muestra la preview con la URL.

○	Si el usuario confirma, hace POST /posts con:

{
  "tempImageId": "tmp-12345",
  "description": "...",
  "filter": "ups_logo"
}
5.	
6.	post-service:

○	ImageService.moveTempToPost(tempImageId) mueve la imagen de temp/ a posts/.

○	Crea el post en Firestore.

○	Devuelve el PostResponse.

________________________________________
6. Resumen de lo que debe quedar en el doc de PyCUDA
Puedes dejar la sección así en tu README de backend / doc técnico:
1.	Descripción del PyCUDA Service.

2.	Endpoint principal: POST /filters/{filterName} (request/respuesta).

3.	Tabla de filtros con nombre, tipo y descripción.

4.	Arquitectura interna:

○	app.py

○	api/filters_controller.py

○	core/filter_service.py

○	core/kernels/*.cu

○	utils/image_io.py

5.	Flujo de datos:

○	App → Gateway → post-service → PyCUDA → Storage → Firestore → App.

○	Diferencia entre estrategia simple y estrategia con preview.

