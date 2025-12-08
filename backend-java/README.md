
Salta al contenido principal
Grado 67
Área personal
Mis cursos
57
391584
Examenes
Proyecto de Interciclo

Proyecto de Interciclo
Requisitos de finalización
Apertura: miércoles, 3 de diciembre de 2025, 00:00
Cierre: miércoles, 10 de diciembre de 2025, 14:00
UPSGlam 2.0 - Plataforma Social con Procesamiento GPU, Firebase y Microservicios Reactivos

Descripción General

El proyecto UPSGlam 2.0 busca desarrollar una plataforma social de imágenes tipo Instagram, combinando tecnologías de computación paralela (CUDA), arquitectura de microservicios reactivos (WebFlux) y servicios en la nube (Firebase). El objetivo es ofrecer una experiencia moderna y escalable, donde los estudiantes aplicarán conceptos de programación paralela, computación reactiva y despliegue con contenedores.

El proyecto estará compuesto por tres pilares:

Procesamiento de imágenes con CUDA (Python)
Gestión de usuarios y publicaciones con Firebase + WebFlux (Java)
Aplicación móvil Flutter con integración en tiempo real
Actividades Principales

1. Procesamiento de Imágenes con CUDA

Implementar al menos seis filtros de convolución personalizados:
4 filtros vistos en clase.
2 nuevos filtros creativos (al menos uno con el logo de UPS o elementos representativos de la UPS).
Usar PyCUDA para el procesamiento paralelo de imágenes en GPU.
Exponer el procesamiento a través de un servicio REST consumible por otros módulos.
2. Backend Reactivo con WebFlux + Firebase

Desarrollar una API Gateway y microservicios con Spring WebFlux aplicando principios de concurrencia y no-bloqueo.
Utilizar Firebase Authentication para la gestión de usuarios y autenticación.
Almacenar publicaciones, likes y comentarios en Firebase Firestore (NoSQL).
Implementar endpoints REST con Webflux para:
Registro/login.
CRUD de publicaciones.
Gestión de procesamiento de imágenes
Implementar consultas Firebase para:
Likes y comentarios en tiempo real.
3. Carga y Distribución de Imágenes con Firebase Hosting

Almacenar imágenes procesadas en Firebase Storage.
Exponer públicamente las imágenes mediante URLs accesibles desde la App móvil.
4. Aplicación Móvil UPSGlam 2.0

Desarrollar una app móvil con Flutter/Ionic/Android:
Registro e inicio de sesión.
Publicación de imágenes con selección de filtros.
Visualización de un feed en orden cronológico.
Likes y comentarios en tiempo real.
Configuración de URL de la API.
UX/UI moderno y enfocado en la experiencia de usuario.
5. Integración y Dockerización Completa

Crear un Docker Compose que integre:
Servicio de convolución (PyCUDA).
API reactiva (WebFlux + Firebase).
Configurar una red común para permitir la comunicación entre contenedores.
6. Demostración y Presentación Final

Demostrar el funcionamiento de la plataforma en un entorno local con todos los servicios corriendo en contenedores.
Realizar una presentación explicativa del flujo de trabajo, arquitectura y resultados obtenidos.
Entregables

Repositorio GitHub con:
Código fuente de cada componente (CUDA, API WebFlux, App).
Dockerfile(s) y archivos de configuración de Docker Compose.
Documentación técnica en formato README.md (arquitectura, despliegue, uso).
APK de la app móvil.
Reporte final con reflexión sobre el uso de tecnologías paralelas, reactivas y en la nube.
Observaciones

Se valorará la creatividad en los filtros de convolución y su relación con la identidad UPS.
El uso adecuado de programación reactiva, concurrencia y despliegue con Docker será parte fundamental de la evaluación.
Se espera un proyecto funcional, no solo prototípico, que simule condiciones reales de uso en red local.
Rúbrica de Evaluación - Proyecto UPSGlam 2.0

(20 puntos)

Criterio

90% - 100%

75% - 89%

60% - 74%

45% - 59%

0% - 44%

1. Procesamiento de Imágenes con CUDA (5 pts)

Implementación completa y optimizada de los filtros de convolución, incluyendo los 3 nuevos filtros creativos propuestos. Funcionamiento correcto, eficiente y documentado. Uso adecuado de PyCUDA aprovechando la GPU.

Implementación adecuada con funcionamiento correcto de la mayoría de los filtros, pero con margen de optimización. Documentación aceptable.

Implementación básica con algunos filtros limitados o errores menores. Documentación insuficiente.

Implementación deficiente, con errores importantes en filtros o mal uso de PyCUDA. Documentación escasa.

Implementación incorrecta o no realizada. Filtros no funcionan o no se implementaron.

2. Backend Reactivo con WebFlux + Firebase (5 pts)

Desarrollo completo y eficiente de la API con WebFlux y Firebase (Auth y Firestore). Manejo adecuado de concurrencia, validaciones y errores. Integración correcta de likes, comentarios y feed en tiempo real.

API funcional con WebFlux y Firebase, aunque con algunas deficiencias en manejo de concurrencia o validaciones. Funcionalidades completas pero con detalles por mejorar.

API básica con funcionalidades parciales, errores menores y uso limitado de programación reactiva.

API deficiente con errores significativos, sin aprovechar la programación reactiva. Funcionalidades incompletas.

API no desarrollada o sin integración con Firebase ni WebFlux.

3. Carga de Imágenes en Firebase Hosting (2 pts)

Imágenes procesadas correctamente cargadas en Firebase Storage y distribuidas mediante URLs accesibles y optimizadas.

Carga funcional de imágenes en Firebase Storage con detalles menores por corregir en accesibilidad o rendimiento.

Carga básica de imágenes con errores menores o sin optimización.

Carga de imágenes con errores significativos o funcionamiento deficiente.

No se realizó la carga de imágenes en Firebase Hosting.

4. Aplicación Móvil (4 pts)

App móvil completa y funcional con todas las características (registro, publicaciones, likes, comentarios, filtros, configuración de API). Interfaz atractiva y experiencia de usuario fluida.

App móvil funcional pero con áreas de mejora en interfaz o experiencia de usuario. Todas las funcionalidades implementadas.

App básica con funcionalidades limitadas o errores menores. Interfaz simple.

App con errores significativos y funcionalidades incompletas. Interfaz deficiente.

App no desarrollada o no funcional.

5. Integración con Docker Compose (2 pts)

Integración completa de los servicios (PyCUDA, API WebFlux, Firebase Emulator si aplica) en un entorno Docker Compose funcional y documentado.

Integración adecuada en Docker Compose con pequeñas mejoras posibles.

Integración básica con errores menores en la configuración de contenedores.

Integración deficiente con errores importantes en la red o configuración.

No se realizó la integración con Docker Compose.

6. Presentación y Demostración en Red Local (2 pt)

Demostración exitosa del proyecto con todos los servicios comunicándose en red local, explicando arquitectura y flujos correctamente.

Demostración funcional con detalles menores por mejorar en presentación o explicación.

Demostración básica con errores menores de integración.

Demostración deficiente con fallos importantes en la integración en red local.

No se realizó la demostración del proyecto.

 

 
