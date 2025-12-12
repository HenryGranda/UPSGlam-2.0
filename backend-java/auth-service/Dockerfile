# ===========================
# STAGE 1: Build con Maven
# ===========================
FROM maven:3.9-eclipse-temurin-21 AS builder

WORKDIR /app

# Copiamos pom y src para compilar
COPY pom.xml .
COPY src ./src

# Compilamos el proyecto y generamos el .jar
RUN mvn -q clean package -DskipTests

# ===========================
# STAGE 2: Imagen ligera para runtime
# ===========================
FROM eclipse-temurin:21-jre

WORKDIR /app

# Copiamos el jar generado desde el builder
COPY --from=builder /app/target/auth-service-*.jar app.jar

# Puerto donde arranca el auth service
ENV SERVER_PORT=8082
EXPOSE 8082

# Variables de entorno para Firebase (se pueden sobreescribir en docker-compose)
ENV FIREBASE_PROJECT_ID=""

# Configuración de Spring Profile (docker por defecto)
ENV SPRING_PROFILES_ACTIVE=docker

# Variables extra para tuning de memoria
ENV JAVA_OPTS="-Xmx512m -Xms256m"

# Comando de arranque con perfil activo
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar --spring.profiles.active=${SPRING_PROFILES_ACTIVE}"]
