# 🌊 RES-Q: Sistema de Soporte a la Decisión Espacial y Optimización Logística Cuántica (COER Piura)

**Equipo Res-Q:**
* Cristian Antony Medina Vilca
* Diego Emerson Huayta Cordova
* Ayrthon Nevy Mamani Mamani
* Ghiis Kiara Soto Godoy

---

## 📌 Descripción del Problema
El departamento de Piura concentra cerca de la tercera parte del impacto humanitario nacional ante eventos hidrometeorológicos extremos como el Fenómeno de El Niño Costero (2017, 2023), el Ciclón Yaku y eventos sísmicos (Sullana 2021). Durante las **primeras 72 horas (ventana crítica de supervivencia)**, hasta el 80% de los 1,200 km de la red vial departamental sufre interrupciones críticas.

**RES-Q** resuelve simultáneamente el **Problema de Localización de Instalaciones con Capacidad (CFLP)** y el **Problema de Localización de Máxima Cobertura (MCLP)** bajo incertidumbre vial mediante computación cuántica variacional (**QAOA** y **VQE** sobre Qiskit).

---

## 🗺️ Escenarios Estocásticos Modelados ($G(S) = (V, E_s, W_s)$)
1. **Escenario $s_1$ (Ruptura del Eje Alto Piura):** Colapso o cierre de los puentes Independencia y Carrasquillo en el corredor Piura - Chulucanas - Morropón (PE-1NR).
2. **Escenario $s_2$ (Aislamiento de la Sierra Piurana):** Huaicos y aluviones en Pusmalca y Agua Blanca en la ruta Empalme PE-1NJ - Canchaque - Huancabamba.
3. **Escenario $s_3$ (Inundación del Bajo Piura):** Desborde del río Piura ($Q > 3,000\text{ m}^3/\text{s}$) y rotura de diques en la Vía de Evitamiento y eje Piura - Catacaos - Cura Mori (PE-1N).

---

## ⚛️ Arquitectura del Sistema
```text
res-q/
├── config/piura_network.json   # Grafo estocástico de Piura, demandas BAH y escenarios s1, s2, s3
├── src/
│   ├── geo/                    # Paso 1: Ingesta geoespacial, Dijkstra y poda topológica de qubits
│   ├── quantum/                # Pasos 2 y 3: Compilador QUBO, Hamiltoniano de Ising, QAOA y VQE
│   ├── classical/              # Paso 4: Benchmarking contra ILP (Branch & Bound), SA y Greedy
│   ├── postprocess/            # Paso 4: Decodificación de bitstrings y validación de ventana 72h
│   └── ui/                     # Paso 5: Dashboard Cartográfico COER Piura (Streamlit + Folium)
├── tests/                      # Suite de pruebas unitarias y de integración
└── app.py                      # Punto de entrada de la plataforma interactiva