# Detección de mensajes SMS spam mediante fine-tuning de FLAN-T5-small

Proyecto de fin de parcial — Segundo Parcial · **Redes Neuronales y Aprendizaje Profundo**

Adaptación de un modelo pre-entrenado (`google/flan-t5-small`) al problema de clasificación de mensajes SMS como **ham** (legítimo) o **spam**, comparando cuantitativamente el modelo base evaluado en *zero-shot* contra el modelo con *full fine-tuning*.

**Paper de referencia:** Labonne, M., & Moran, S. (2023). *Spam-T5: Benchmarking Large Language Models for Few-Shot Email Spam Detection*. arXiv. https://arxiv.org/abs/2304.01238

**Dataset:** SMS Spam Collection — https://huggingface.co/datasets/ucirvine/sms_spam (se descarga automáticamente con código en el notebook; no se incluyen archivos de datos en el repositorio).

## Resultados principales (antes vs. después)

Evaluación sobre el mismo conjunto de prueba (774 mensajes, partición estratificada 70/15/15, semilla 42):

| Métrica | Modelo base (zero-shot) | Modelo fine-tuneado | Diferencia |
|---|---|---|---|
| Accuracy | 0.164 | 0.997 | +0.833 |
| Precision (spam) | 0.129 | 0.990 | +0.860 |
| Recall (spam) | 1.000 | 0.990 | −0.010 |
| F1 (spam) | 0.229 | 0.990 | +0.761 |
| F1 macro | 0.158 | 0.994 | +0.836 |
| Respuestas inválidas | 0 | 0 | 0 |

El modelo base en zero-shot presenta un sesgo severo hacia la clase `spam` (responde "spam" casi siempre: recall perfecto con precisión de 0.13). El fine-tuning sobre ~3,600 ejemplos corrige ese sesgo y deja únicamente 2 errores en el conjunto de prueba. El análisis completo está en las secciones 6 y 7 del notebook.

## Estructura del repositorio

```
.
├── PROYECTO_Final_RNAP.ipynb    # Notebook completo (secciones 1-8)
├── informe/
│   └── Informe_RNAP.pdf         # Informe académico (máx. 12 páginas)
├── README.md
├── DECLARACION_USO_IA.md        # Declaración de uso de IA
└── .gitignore
```

> El modelo entrenado (`flan_t5_spam_finetuned/`) se genera al ejecutar el notebook y **no** se sube al repositorio . Tampoco se incluyen archivos del dataset: el notebook lo descarga con código. Contenido de `.gitignore`:
>
> ```
> flan_t5_spam_finetuned/
> *.bin
> *.safetensors
> .ipynb_checkpoints/
> ```

## Cómo ejecutar

1. Abrir `PROYECTO_Final_RNAP.ipynb` en **Google Colab**.
2. Activar entorno con GPU: *Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)*.
3. Ejecutar *Entorno de ejecución → Reiniciar y ejecutar todo* (**Restart & Run All**).

Las dependencias (`datasets`, `transformers`, `accelerate`, `sentencepiece`, `evaluate`, `scikit-learn`) se instalan desde el propio notebook. El dataset se descarga automáticamente desde Hugging Face. Tiempos de referencia con GPU T4: inferencia zero-shot ≈ 6 s; fine-tuning (3 épocas) ≈ pocos minutos.

**Reproducibilidad:** semilla fija (`SEED = 42`) en la división estratificada, en PyTorch y en el entrenador; `fp16` desactivado (T5 fue pre-entrenado en `bfloat16` y colapsa a `NaN` con `float16`); selección del mejor checkpoint por `eval_loss`.

## Correspondencia con la estructura exigida en la guía

| Sección exigida | Ubicación en el notebook |
|---|---|
| § 1 Paper y problema | Sección 1 |
| § 2 Dataset y EDA | Sección 2 |
| § 3 Implementación de la adaptación / fine-tuning | Secciones 3 (baseline zero-shot) y 4 (fine-tuning) |
| § 4 Evaluación: antes y después | Secciones 3.7–3.9 (antes) y 5 (después + comparación) |
| § 5 Análisis crítico | Sección 6 |
| § 6 Conclusiones | Sección 7 |

La sección 8 del notebook es adicional: demostración de inferencia en vivo con medida de confianza (softmax sobre los logits de `ham`/`spam`), umbral de abstención de 0.90 para revisión manual, y descripción del proceso de despliegue como servicio real.
| *(Extensión práctica)* | Sección 8 — prueba del modelo en vivo con confianza/abstención y proceso de despliegue |

> Los valores numéricos de las secciones 6 y 7 se generan dinámicamente a partir de los resultados calculados en el notebook (no hay cifras escritas a mano), por lo que permanecen consistentes tras cada re-ejecución.

## Metodología resumida

1. **EDA y limpieza:** 5,574 mensajes originales → 5,160 tras eliminar duplicados/vacíos (87.6 % ham / 12.4 % spam). División estratificada 70/15/15.
2. **Baseline (antes):** FLAN-T5-small sin entrenar, evaluado con el prompt `Classify the following SMS message as spam or ham: ... Answer only spam or ham:` y normalización de respuestas a `ham` / `spam` / `invalid`.
3. **Adaptación:** *full fine-tuning* texto-a-texto (3 épocas, LR 3e-4, batch 16, weight decay 0.01, AdamW).
4. **Evaluación (después):** mismas métricas sobre el mismo split de prueba, matrices de confusión, tabla comparativa, ejemplos corregidos y errores restantes.

## Integrantes

| Integrante | Rol |
|---|---|
| Arianna Faustos | Paper, problema y fundamentos |
| Ronny Oswaldo Guamán | Dataset y análisis exploratorio |
| Erick Cabrera | Modelo base zero-shot |
| Carlos Andrés Anchundia Toala | Fine-tuning del modelo |
| Jair jander loor Holguín | Evaluación y comparación |
| Fernando Baque | Integración, análisis crítico, informe y GitHub |

