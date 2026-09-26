# Actividad 09 (Unidad II) — PCA, NMF y detección de anomalías

**Universidad Nacional del Altiplano - Puno** · Ingeniería de Sistemas · Maestría en Ciencia de Datos

**Curso:** Machine Learning I - Grupo A

**Presentado por:**
- Darwin Rigoberto Mamani Quispe
- Melbis Chino Huaycani

📓 Notebook completo: [`Actividad_09.ipynb`](./Actividad_09.ipynb)

---

## Resumen Ejecutivo

La actividad desarrolla tres aplicaciones de aprendizaje no supervisado: reconocimiento facial mediante **Eigenfaces (PCA)**, recomendación de artículos mediante **NMF + similitud de coseno** y detección de anomalías en series temporales mediante un **ensemble de KNN, LOF e Isolation Forest**.

## A. Reconocimiento de rostros con Eigenfaces

Se utiliza **Olivetti Faces** de scikit-learn, compuesto por imágenes faciales de 64×64 píxeles. Cada imagen contiene 4096 características. Se divide el conjunto en entrenamiento y prueba y se aplica PCA con 100 componentes.

Los componentes principales pueden visualizarse como imágenes y reciben el nombre de **Eigenfaces**. Después de transformar los rostros al espacio PCA, se entrena un clasificador SVM para identificar a las personas.

El flujo es:

`Olivetti Faces → train/test → PCA (100 componentes) → Eigenfaces → SVM → accuracy`

La reducción permite trabajar con una representación mucho más compacta sin recurrir a redes neuronales. El notebook imprime la varianza explicada, el accuracy y el reporte de clasificación, además de mostrar ejemplos de predicciones.

## B. Recomendación de artículos con NMF

Se construye un pequeño conjunto de artículos de cinco áreas: tecnología, deportes, salud, economía y ciencia.

Primero se transforma el texto mediante **TF-IDF**. Después se aplica **NMF con 5 componentes**, donde cada componente representa un tema latente y cada artículo queda expresado como una combinación no negativa de esos temas.

Para recomendar contenido se calcula la **similitud de coseno** entre las representaciones NMF:

`Artículos → TF-IDF → NMF → vectores temáticos → similitud de coseno → Top-N recomendaciones`

El notebook también muestra las palabras con mayor peso de cada componente, lo que permite interpretar los temas encontrados por NMF.

## C. Ensemble para detección de anomalías en series temporales

Se genera una serie temporal diaria reproducible con tendencia, estacionalidad y ruido, a la que se agregan seis anomalías conocidas.

Se calculan cuatro características: valor, diferencia diaria, media móvil de 7 días y desviación móvil de 7 días. Sobre estas variables se aplican tres detectores:

| Detector | Criterio |
|---|---|
| KNN | Distancia a vecinos cercanos |
| LOF | Densidad local |
| Isolation Forest | Facilidad de aislamiento |

Cada algoritmo genera una puntuación en una escala diferente. Por ello los scores se normalizan al intervalo `[0,1]` y se promedian:

`ensemble_score = (score_KNN + score_LOF + score_IForest) / 3`

Finalmente se consideran anómalas las observaciones ubicadas en el 2 % superior del `ensemble_score`. Como la serie fue generada con anomalías conocidas, se evalúa el resultado mediante **precision, recall, F1 y matriz de confusión**.

## Conclusión general

PCA, NMF y los métodos de detección de anomalías resuelven problemas distintos, pero comparten la capacidad de descubrir estructura útil sin depender de redes neuronales. PCA resume la información facial mediante componentes principales; NMF obtiene temas interpretables para comparar documentos; y el ensemble combina diferentes criterios de anomalía para obtener una decisión más estable.

La actividad evidencia que la reducción de dimensionalidad no solo disminuye el costo computacional, sino que puede generar representaciones útiles para tareas posteriores. Asimismo, la combinación de detectores resulta conveniente cuando una anomalía puede manifestarse por distancia, baja densidad o facilidad de aislamiento.

---

## Ejecución

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Luego abrir y ejecutar todas las celdas de `Actividad_09.ipynb`.

> La primera ejecución de `fetch_olivetti_faces()` puede descargar el dataset Olivetti Faces si todavía no está disponible en la caché local de scikit-learn.
