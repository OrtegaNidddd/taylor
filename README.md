# 📐 Visualizador Interactivo del Polinomio de Taylor en $\mathbb{R}^2$

Herramienta web educativa e interactiva diseñada para la asignatura de **Cálculo Multivariado**. Conecta el rigor analítico y algebraico del Polinomio de Taylor en dos variables con su interpretación geométrica y espacial en 3D en tiempo real, e integra un **Tutor Inteligente con IA** (Gemini y DeepSeek).

---

## 🚀 Características Principales

- **Entrada de Funciones Analíticas Bidimensionales:** Parser matemático seguro que admite funciones elementales (`sin`, `cos`, `exp`, `log`, `sqrt`, etc.), multiplicación implícita (`2x`, `3y`) y potencias con `^`.
- **Motor Simbólico con SymPy:**
  - Cálculo general de Taylor para cualquier grado $n \ge 1$.
  - Plano tangente ($n=1$) y gradiente $\nabla f(x_0, y_0)$.
  - Matriz Hessiana y análisis de curvatura ($n \ge 2$).
  - Desglose término a término por órdenes con valores simbólicos y numéricos.
- **Evaluación Numérica Vectorizada con NumPy:** Discretización espacial sobre mallas 2D con manejo seguro de discontinuidades y filtrado de singularidades.
- **Visor Tridimensional Interactivo con Plotly:**
  - Superficies diferenciadas por paletas contrastantes y opacidad ajustable.
  - Marcador tridimensional del punto de apoyo $(x_0, y_0, f(x_0, y_0))$.
  - Identificación visual de la **zona de convergencia / tolerancia** ($|f - P_n| \le \epsilon$).
  - Superficie dedicada opcional para el residuo absoluto.
  - Rotación orbital libre, paneo, zoom y lectura de coordenadas al pasar el cursor.
- **🤖 Tutor Inteligente con IA (Gemini & DeepSeek):**
  - Generación de explicaciones pedagógicas adaptadas al estado actual de la función y su residuo.
  - Capacidad para responder preguntas libres de los alumnos sobre el comportamiento geométrico.
  - Diagnóstico analítico estructurado automático incluso en modo offline / sin API Key.
- **Didáctica y Reactividad con Streamlit:** Actualización inmediata ante cambios de parámetros, tarjetas de métricas (RMSE, % de cobertura válida) y catálogo de funciones de estudio guiadas.

---

## 🔑 Configuración de Variables de Entorno (Tutor IA)

El proyecto incluye soporte para **Google Gemini** (con nivel gratuito en [Google AI Studio](https://aistudio.google.com/)) y **DeepSeek** ([DeepSeek Platform](https://platform.deepseek.com/)).

1. Copia o edita el archivo `.env` en la raíz del proyecto:
   ```bash
   cp .env.example .env
   ```

2. Agrega tu clave en el archivo `.env`:
   ```ini
   # Proveedor por defecto: "gemini" o "deepseek"
   AI_PROVIDER=gemini

   # Clave de Google Gemini (Gratis en https://aistudio.google.com/)
   GEMINI_API_KEY=tu_api_key_aqui
   GEMINI_MODEL=gemini-1.5-flash

   # Clave de DeepSeek (opcional)
   DEEPSEEK_API_KEY=tu_api_key_aqui
   DEEPSEEK_MODEL=deepseek-chat
   ```

> 💡 *También puedes ingresar tu API Key directamente en el campo de texto de la barra lateral de la aplicación web.*

---

## 📦 Instalación y Requisitos

Requiere **Python 3.10+**.

1. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Ejecución de la Aplicación

### Opción A: Ejecución Local Directa

```bash
streamlit run app.py
```
La aplicación se abrirá en `http://localhost:8501`.

---

### Opción B: Despliegue con Docker y Docker Compose (Recomendado para Producción)

1. **Construir y levantar el contenedor en segundo plano:**
   ```bash
   docker compose up -d --build
   ```

2. **Verificar el estado y logs:**
   ```bash
   docker compose ps
   docker compose logs -f
   ```

3. **Detener la aplicación:**
   ```bash
   docker compose down
   ```

#### Ejecución alternativa con Docker CLI manual:
```bash
# 1. Construir la imagen
docker build -t visualizador-taylor .

# 2. Ejecutar el contenedor inyectando el archivo .env
docker run -d -p 8501:8501 --env-file .env --name visualizador_app visualizador-taylor
```

---

## 🧪 Ejecución de Pruebas Automatizadas

El proyecto cuenta con una suite completa de pruebas unitarias y de rendimiento:

```bash
pytest -v
```

---

## 📂 Estructura del Código

```text
visualizador_taylor/
├── app.py                     # Entrypoint de Streamlit y coordinación
├── core/
│   ├── __init__.py
│   ├── ai_tutor.py            # Tutor pedagógico con IA (Gemini y DeepSeek)
│   ├── parser.py              # Validación, sanitización y parseo seguro
│   ├── taylor_engine.py       # Algoritmo de Taylor multivariable (SymPy)
│   └── numeric_eval.py        # Malla NumPy, cálculo de residuo y tolerancia
├── ui/
│   ├── __init__.py
│   ├── components.py          # Panel lateral, controles y catálogo de presets
│   ├── latex_display.py       # Renderizado de fórmulas y pasos en LaTeX
│   └── plotting_3d.py         # Generación de gráficos espaciales Plotly
├── tests/
│   ├── test_ai_tutor.py       # Pruebas del tutor inteligente
│   ├── test_parser.py         # Pruebas de seguridad y sintaxis
│   ├── test_taylor.py         # Pruebas de exactitud analítica de polinomios
│   └── test_numeric.py        # Pruebas de mallas, dimensiones y rendimiento
├── .env                       # Archivo local de variables de entorno y API keys
├── .env.example               # Plantilla de variables de entorno de ejemplo
├── PLAN_DE_TRABAJO.md         # Plan de trabajo detallado y checklist de avance
├── requirements.txt           # Dependencias fijadas del proyecto
└── README.md                  # Documentación del sistema
```
