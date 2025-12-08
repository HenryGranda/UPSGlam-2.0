# 🔒 Security & Credentials Setup

## ⚠️ IMPORTANTE: Archivos Sensibles

Los siguientes archivos contienen credenciales y **NUNCA deben ser commiteados**:

```
❌ NO COMMITEAR:
- src/main/resources/application-local.yml
- src/main/resources/firebase-credentials.json
- target/ (toda la carpeta)
- *.jar, *.war
```

✅ Estos archivos están protegidos en `.gitignore`

---

## 📝 Setup Inicial

### 1. Firebase Credentials

1. Ve a [Firebase Console](https://console.firebase.google.com)
2. Selecciona tu proyecto `upsglam-8c88f`
3. **Project Settings** → **Service Accounts**
4. Click en **Generate New Private Key**
5. Descarga el archivo JSON
6. **Renombra** el archivo a: `firebase-credentials.json`
7. **Coloca** en: `src/main/resources/firebase-credentials.json`

```bash
# Estructura del archivo:
{
  "type": "service_account",
  "project_id": "upsglam-8c88f",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...",
  "client_email": "firebase-adminsdk@upsglam-8c88f.iam.gserviceaccount.com",
  ...
}
```

### 2. Application Local Configuration

1. Copia el archivo de ejemplo:
```bash
cp src/main/resources/application-local.yml.example src/main/resources/application-local.yml
```

2. Edita `application-local.yml` con tus credenciales:

```yaml
firebase:
  api-key: AIzaSy... # Tu API Key de Firebase

supabase:
  url: https://xxxxx.supabase.co # URL de tu proyecto
  key: eyJhbGci... # Anon public key
  service-role-key: eyJhbGci... # Service role key (SECRETO)
  storage:
    bucket: upsglam
    folders:
      posts: posts
      temp: temp
      avatars: avatars
```

### 3. Supabase Setup

Ver guía completa en: [`SUPABASE-SETUP.md`](SUPABASE-SETUP.md)

**Resumen rápido:**
1. Crea proyecto en [Supabase](https://supabase.com/dashboard)
2. Storage → Create bucket "upsglam" (público)
3. Crea carpetas: `posts/`, `temp/`, `avatars/`
4. Copia credenciales de Settings → API

---

## 🚨 Si Commiteaste Credenciales

Si accidentalmente commiteaste archivos sensibles:

### Opción 1: Remover del último commit (antes de push)
```bash
git reset HEAD~1
# Edita .gitignore para incluir el archivo
git add .gitignore
git commit -m "Add gitignore for sensitive files"
```

### Opción 2: Remover del tracking (ya fue pushed)
```bash
# Remover del tracking pero mantener archivo local
git rm --cached src/main/resources/application-local.yml
git rm --cached src/main/resources/firebase-credentials.json

# Commit los cambios
git commit -m "Remove sensitive credentials from tracking"
git push

# IMPORTANTE: Rotar las credenciales comprometidas
# - Firebase: Revocar key y generar nueva
# - Supabase: Regenerar service role key
```

### Opción 3: Limpiar historial completo (última opción)
⚠️ **Solo si las credenciales están en múltiples commits**

```bash
# Usar git-filter-repo (instalar primero)
pip install git-filter-repo

# Remover archivo de todo el historial
git filter-repo --path src/main/resources/application-local.yml --invert-paths
git filter-repo --path src/main/resources/firebase-credentials.json --invert-paths

# Force push (⚠️ coordinar con el equipo)
git push --force
```

---

## ✅ Checklist de Seguridad

Antes de commitear, verifica:

- [ ] `application-local.yml` NO está en git
- [ ] `firebase-credentials.json` NO está en git
- [ ] `.gitignore` incluye archivos sensibles
- [ ] `application-local.yml.example` SÍ está en git (sin credenciales)
- [ ] Ningún archivo en `target/` está trackeado
- [ ] No hay archivos `.jar` o `.war` en git

**Comando para verificar:**
```bash
git status --ignored
```

---

## 🔑 Variables de Entorno (Producción)

Para producción, usa variables de entorno en lugar de archivos:

```bash
# Environment variables
export FIREBASE_PROJECT_ID=upsglam-8c88f
export FIREBASE_CREDENTIALS_JSON=$(cat /path/to/firebase-credentials.json)
export SUPABASE_URL=https://xxxxx.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=eyJhbGci...

# Luego en application-prod.yml:
firebase:
  project-id: ${FIREBASE_PROJECT_ID}
  credentials-json: ${FIREBASE_CREDENTIALS_JSON}

supabase:
  url: ${SUPABASE_URL}
  service-role-key: ${SUPABASE_SERVICE_ROLE_KEY}
```

---

## 📚 Recursos

- [Firebase Security Best Practices](https://firebase.google.com/docs/rules/basics)
- [Supabase Security](https://supabase.com/docs/guides/security)
- [Git Remove Sensitive Data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)

---

**Última actualización:** 7 de diciembre de 2025
