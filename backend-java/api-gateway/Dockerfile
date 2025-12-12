# ===========================
# STAGE 1: Build con Maven
# ===========================
FROM maven:3.9-eclipse-temurin-21 AS builder

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos pom y src para compilar
COPY pom.xml .
COPY src ./src

# Compilamos el proyecto y generamos el .jar
RUN mvn -q clean package -DskipTests

# ===========================
# STAGE 2: Imagen ligera para runtime
# ===========================
FROM eclipse-temurin:21-jre-alpine

WORKDIR /app

# Copiamos el jar generado desde el builder
COPY --from=builder /app/target/api-gateway-*.jar app.jar

# Puerto donde arranca el gateway
ENV SERVER_PORT=8080
EXPOSE 8080

# Variables extra por si luego quieres tunear memoria, etc.
ENV JAVA_OPTS=""

# Comando de arranque
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
