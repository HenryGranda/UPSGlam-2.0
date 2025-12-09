# 🔐 Configuración de Credenciales

## ⚠️ IMPORTANTE: Seguridad de Credenciales

Este proyecto requiere credenciales sensibles que **NUNCA deben ser commiteadas** a Git.

## 📋 Archivos Protegidos

Los siguientes archivos contienen credenciales y están excluidos en `.gitignore`:

- `**/application-local.yml` - Credenciales de Firebase y Supabase
- `**/start-post.ps1` - Variables de entorno de Supabase
- `**/firebase-credentials.json` - Service Account de Firebase

## 🚀 Configuración Inicial

### 1. Post Service

#### a) Configurar application-local.yml

```bash
cd backend-java/post-service/src/main/resources/
cp application-local.yml.example application-local.yml
```

Edita `application-local.yml` y reemplaza:
- `YOUR_FIREBASE_API_KEY_HERE` → Tu Firebase API Key
- `YOUR_PROJECT_ID.supabase.co` → URL de tu proyecto Supabase
- `YOUR_ANON_KEY_HERE` → Supabase anon/public key
- `YOUR_SERVICE_ROLE_KEY_HERE` → Supabase service_role key

#### b) Configurar start-post.ps1

```bash
cd backend-java/post-service/docs/
cp start-post.ps1.example start-post.ps1
```

Edita `start-post.ps1` y reemplaza las credenciales de Supabase.

### 2. Auth Service

```bash
cd backend-java/auth-service/src/main/resources/
cp application-local.yml.example application-local.yml
```

Edita y reemplaza `YOUR_FIREBASE_API_KEY_HERE`.

### 3. Firebase Service Account

1. Ve a Firebase Console → Project Settings → Service Accounts
2. Genera una nueva private key
3. Guarda el archivo JSON como:
   - Post Service: `backend-java/post-service/src/main/resources/firebase-credentials.json`
   - Auth Service: `backend-java/auth-service/src/main/resources/firebase-credentials.json`

## 🔑 Dónde Obtener las Credenciales

### Firebase
1. Ve a [Firebase Console](https://console.firebase.google.com/)
2. Selecciona tu proyecto
3. **API Key**: Project Settings → General → Web API Key
4. **Service Account**: Project Settings → Service Accounts → Generate New Private Key

### Supabase
1. Ve a [Supabase Dashboard](https://app.supabase.com/)
2. Selecciona tu proyecto
3. Settings → API
   - **URL**: Project URL
   - **anon key**: anon public
   - **service_role key**: service_role (⚠️ Mantener secreto)

## 🛡️ Si Expusiste Credenciales Accidentalmente

### 1. Revocar Inmediatamente
- **Supabase**: Dashboard → Settings → API → Reset Keys
- **Firebase**: Console → Project Settings → Regenerate API Key

### 2. Remover del Historial de Git

```bash
# Remover del tracking (mantiene archivo local)
git rm --cached path/to/sensitive-file.yml

# Limpiar del historial (⚠️ Reescribe historia)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/sensitive-file.yml" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (coordinar con el equipo)
git push origin --force --all
```

### 3. Usar BFG Repo-Cleaner (Recomendado)

```bash
# Instalar BFG
# https://rtyley.github.io/bfg-repo-cleaner/

# Limpiar credenciales
java -jar bfg.jar --delete-files application-local.yml
java -jar bfg.jar --delete-files firebase-credentials.json
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

## ✅ Verificación

Antes de cada commit, verifica:

```bash
# Ver qué archivos están staged
git status

# Verificar que no haya credenciales
git diff --cached | grep -i "api.key\|apikey\|secret\|password"
```

## 📚 Referencias

- [GitHub: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [Supabase Security Best Practices](https://supabase.com/docs/guides/platform/security)
- [Firebase Security](https://firebase.google.com/docs/projects/api-keys)

## 🤝 Para el Equipo

- **NUNCA** compartas credenciales por chat, email o docs públicos
- Usa `.env` local o gestores de secretos (AWS Secrets Manager, Azure Key Vault)
- Para producción, usa variables de entorno del servidor
- Rota las credenciales regularmente
