# Guía de Despliegue en Hugging Face Spaces 🚀

Hugging Face Spaces es ideal para aplicaciones de Machine Learning porque su **Tread Gratuito (Basic)** te proporciona **16 GB de RAM y 2 vCPUs**, lo cual evitará que el entorno colapse al ejecutar los modelos de `xgboost` y los backtests de `vectorbt`.

Sigue estos sencillos pasos para tener tu Data Investment Hub online de forma gratuita y profesional.

## Pasos para Desplegar

### 1. Preparar la estructura en GitHub o Local
Asegúrate de que la raíz de la carpeta que vas a subir contiene los siguientes archivos esenciales:
- `dashboard.py` (Tu archivo principal de Streamlit)
- `finance_ingestor.py`
- `ml_engine.py`
- `backtester.py`
- `daily_bot.py`
- `requirements.txt` (MUY IMPORTANTE para Hugging Face)

### 2. Crear una Cuenta y el Space
1. Ve a [Hugging Face](https://huggingface.co/) y crea una cuenta si no la tienes.
2. Haz clic en tu perfil arriba a la derecha y selecciona **"New Space"**.
3. Rellena los datos de tu Space:
   - **Space name:** `Data-Investment-Hub` (o el nombre que prefieras).
   - **License:** Elige la que corresponda (ej. MIT o Openrail).
   - **Select the Space SDK:** Aritmética. ¡Muy importante! Selecciona **Streamlit**.
   - **Space hardware:** Selecciona el gratuito que te da 2 vCPU y 16 GB Ram (por defecto).
   - **Visibility:** Public.
4. Haz clic en **Create Space**.

### 3. Configurar los Secretos / Variables de Entorno
Al igual que usas un archivo `.env` en local, en Hugging Face necesitamos configurar las claves API (como la de Groq y Telegram) para que sean secretas.

1. Ve a la pestaña **"Settings"** dentro de tu nuevo Space.
2. Desplázate hacia abajo hasta la sección **Variables and secrets**.
3. Añade como **Secrets** (Claves secretas, importante que sea en la sección Secrets y no variables públicas):
   - `GROQ_API_KEY`: *[Tu clave de Groq]*
   - `BOT_TOKEN`: *[Tu token de Telegram]*
   - `BOT_ID`: *[Tu ID de chat]*

**Nota sobre el código:** No tienes que cambiar el código de tu aplicación. El paquete `python-dotenv` (si encuentra el `.env` local lo usará, si está en Hugging Face o Streamlit Cloud, leerá las variables de entorno de allí automáticamente usando `os.getenv()`).

### 4. Subir el Código al Space

Hugging Face Spaces detrás de cámara es un repositorio Git.

**Forma Rápida (Sugerida para empezar):**
1. Ve a la pestaña **"Files"** de tu Space.
2. Usa el botón **"Add file"** > **"Upload files"**.
3. Selecciona y sube arrastrando todos tus scripts Python (`.py`) y tu `requirements.txt`.
4. El archivo principal debe llamarse preferiblemente `app.py`. Si el tuyo se llama `dashboard.py`, puedes renombrarlo temporalmente o cambiar la configuración de Hugging Face para que ejecute `dashboard.py`. **Lo más fácil:** renombrar `dashboard.py` a `app.py` antes de subirlo.

**Forma Avanzada (Con Git):**
Hugging Face te dará las instrucciones de git en la portada de tu Space recién creado:
```bash
git clone https://huggingface.co/spaces/TuUsuario/Data-Investment-Hub
```
Clonas el repo, pegas tus archivos dentro, haces `git add .`, `git commit` y `git push`.

### 5. Configurar el archivo principal (Si no renombraste a app.py)
Si mantienes el nombre `dashboard.py` y subes tus archivos tal cual, necesitas decirle a HF que ese es el archivo de entrada.
1. Crea/edita el archivo `README.md` que Hugging Face genera en tu Space por defecto.
2. En la cabecera (frontmatter), asegúrate de que tiene esto:
```yaml
---
title: Data Investment Hub
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.30.0
app_file: dashboard.py  <-- ¡ESTO ES LO CLAVE!
pinned: false
---
```

### 6. ¡Éxito!
La aplicación comenzará a construirse (estado *"Building"* en amarillo). Tardará un par de minutos en instalar `xgboost` y el resto de las librerías del `requirements.txt`.
Cuando cambie a *"Running"* (verde), ¡tu AI Dashboard Profesional estará online!
