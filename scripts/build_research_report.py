from __future__ import annotations

import csv
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "outputs" / "reports" / "energy_gated_frequency_neuron_report.pdf"


def read_history(run_dir: str) -> dict[str, float | str]:
    path = ROOT / "outputs" / run_dir / "metrics" / "synthetic_history.csv"
    if not path.exists():
        return {"run": run_dir, "status": "missing"}

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    best = max(rows, key=lambda row: float(row["val_acc"]))
    final = rows[-1]
    return {
        "run": run_dir,
        "best_val_acc": float(best["val_acc"]),
        "best_epoch": int(float(best["epoch"])),
        "final_val_acc": float(final["val_acc"]),
        "final_val_loss": float(final["val_loss"]),
        "final_val_gate_mean": float(final.get("val_gate_mean", "nan")),
        "final_val_active_bands": float(final.get("val_active_bands", "nan")),
    }


def read_snr(run_dir: str) -> list[dict[str, str]]:
    path = ROOT / "outputs" / run_dir / "metrics" / "snr_sweep_hard.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def add_text_page(
    pdf: PdfPages,
    title: str,
    paragraphs: list[str],
    footer: str = "Energy-Gated Frequency Neuron",
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = 0.94
    ax.text(0.08, y, title, fontsize=20, fontweight="bold", va="top")
    y -= 0.055
    ax.plot([0.08, 0.92], [y, y], color="#222222", linewidth=1.2)
    y -= 0.035

    for paragraph in paragraphs:
        if paragraph.startswith("## "):
            y -= 0.01
            ax.text(0.08, y, paragraph[3:], fontsize=14, fontweight="bold", va="top")
            y -= 0.035
            continue

        if paragraph.startswith("```") and paragraph.endswith("```"):
            text = paragraph[3:-3].strip("\n")
            lines = text.splitlines()
            box_height = min(0.018 * len(lines) + 0.025, 0.30)
            ax.add_patch(
                plt.Rectangle(
                    (0.08, y - box_height),
                    0.84,
                    box_height,
                    facecolor="#f3f3f3",
                    edgecolor="#cccccc",
                    linewidth=0.8,
                )
            )
            ax.text(
                0.10,
                y - 0.015,
                text,
                fontsize=9,
                family="monospace",
                va="top",
                linespacing=1.35,
            )
            y -= box_height + 0.03
            continue

        wrapped = textwrap.wrap(paragraph, width=92)
        for line in wrapped:
            ax.text(0.08, y, line, fontsize=10.5, va="top")
            y -= 0.021
        y -= 0.014

        if y < 0.10:
            ax.text(0.08, 0.04, footer, fontsize=8, color="#666666")
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            y = 0.94

    ax.text(0.08, 0.04, footer, fontsize=8, color="#666666")
    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, columns: list[str], rows: list[list[str]]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.05, 0.94, title, fontsize=18, fontweight="bold", va="top")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.05, 0.12, 0.90, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.25)

    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dde7f2")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#fbfbfb")

    ax.text(0.05, 0.05, "Generated from local experiment outputs.", fontsize=8, color="#666666")
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path, note: str) -> None:
    if not image_path.exists():
        add_text_page(pdf, title, [f"Image not found: {image_path}", note])
        return

    image = Image.open(image_path)
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.05, 0.95, title, fontsize=18, fontweight="bold", va="top")
    ax.text(0.05, 0.90, note, fontsize=9.5, va="top")

    img_ax = fig.add_axes([0.08, 0.08, 0.84, 0.76])
    img_ax.axis("off")
    img_ax.imshow(image)

    pdf.savefig(fig)
    plt.close(fig)


def format_metric(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.3f}"


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    runs = [
        ("hard_fixed", "Hard + fine filter bank + fixed filters"),
        ("hard_learnable", "Hard + fine filter bank + learnable filters"),
        ("hard_contextual", "Hard + learnable filters + contextual gate"),
        ("hard_sparse_gate", "Hard + learnable filters + sparse gate L1=0.005"),
        ("hard_sparse_gate_l005", "Hard + learnable filters + sparse gate L1=0.05"),
        ("hard_sparse_gate_l002_t07", "Sparse gate with active threshold 0.7"),
    ]
    summaries = [(run, label, read_history(run)) for run, label in runs]

    with PdfPages(REPORT_PATH) as pdf:
        add_text_page(
            pdf,
            "Energy-Gated Frequency Neuron",
            [
                "Reporte tecnico del proyecto de portafolio.",
                "Objetivo: documentar desde cero como surge la idea, que principios fisicos y de deep learning la sostienen, como esta estructurada la red neuronal y como se compara con modelos de audio alternativos.",
                "Estado del proyecto: prototipo experimental sobre datos sinteticos controlados. El modelo ya implementa banco de filtros, activacion, energia por banda, gates aprendibles, filtros aprendibles, gate contextual opcional y regularizacion sparse/selectiva.",
                "Fecha de generacion: 2026-07-10.",
            ],
        )

        add_text_page(
            pdf,
            "1. Origen Del Proyecto",
            [
                "El proyecto nace de una pregunta conceptual: si los filtros de circuitos y procesamiento de senales, como pasa baja, pasa alta y pasa banda, pueden usarse como funciones de activacion en redes neuronales.",
                "La respuesta tecnica es que un filtro no es una activacion clasica. Una activacion como ReLU, tanh o GELU opera punto a punto. Un filtro opera sobre una vecindad temporal mediante convolucion, por lo que introduce memoria y estructura temporal, pero por si solo sigue siendo una operacion lineal.",
                "La idea fuerte no es reemplazar ReLU por un filtro, sino construir un bloque neuronal que combine filtrado interpretable, no linealidad, medicion de energia y compuertas aprendibles por banda de frecuencia.",
                "En audio esto tiene sentido porque la senal no es una lista arbitraria de numeros: contiene componentes de frecuencia, energia, fase y dinamica temporal.",
            ],
        )

        add_text_page(
            pdf,
            "2. Principios Aplicados",
            [
                "## Filtros como convoluciones",
                "Un filtro digital se puede escribir como una convolucion y[n] = sum_k h[k] x[n-k]. En una CNN 1D, los kernels convolucionales tambien actuan como filtros aprendibles.",
                "## No linealidad",
                "Si solo se apilan capas lineales, toda la red puede reducirse a una unica transformacion lineal. Por eso el bloque mantiene una activacion phi despues del filtrado.",
                "## Energia por banda",
                "La energia E_k = mean(u_k[n]^2) mide cuanta actividad tiene una banda. Es una magnitud natural en procesamiento de senales y ofrece una lectura interpretable.",
                "## Compuerta por frecuencia",
                "La compuerta g_k decide cuanto de cada banda pasa al clasificador. Esto conecta la lectura fisica de energia con una decision aprendida por la red.",
            ],
        )

        add_text_page(
            pdf,
            "3. Estructura Matematica Del Bloque",
            [
                "Para una banda k, el bloque Energy-Gated Frequency Neuron calcula:",
                """```
u_k[n] = (h_k * x)[n]
z_k[n] = u_k[n] + b_k
a_k[n] = phi(z_k[n])
E_k = mean_n(u_k[n]^2)
g_k = sigmoid(alpha_k * log(1 + E_k) + beta_k)
y_k[n] = g_k * a_k[n]
```""",
                "La version con gate contextual reemplaza gates independientes por una MLP pequena:",
                """```
E = [log(1 + E_1), ..., log(1 + E_K)]
g = sigmoid(MLP(E))
```""",
                "La version V2 sparse agrega una penalizacion a la funcion de perdida:",
                """```
loss = cross_entropy + lambda_gate * mean(gates)
```""",
                "Esta penalizacion busca que el modelo use menos bandas cuando sea posible, favoreciendo una representacion mas selectiva.",
            ],
        )

        add_text_page(
            pdf,
            "4. Arquitectura Implementada",
            [
                "El modelo completo esta organizado como un frontend neuronal interpretable seguido por un clasificador pequeno.",
                """```
Raw waveform [B, 1, T]
  -> EnergyGatedFrequencyNeuron
       -> filter bank
       -> activation
       -> energy estimation
       -> learned gate
  -> FrequencyPooling
       -> mean, max, std, log-energy, gates
  -> MLP classifier
  -> logits [B, num_classes]
```""",
                "El clasificador se mantiene deliberadamente pequeno para que el protagonista sea el bloque fisico-inspirado, no una cabeza de decision demasiado potente.",
                "El banco de filtros puede ser default o fine. El banco fine divide mejor la zona 250-1500 Hz, que fue necesaria para los experimentos hard.",
            ],
        )

        add_text_page(
            pdf,
            "5. Evolucion Experimental",
            [
                "El proyecto empezo con un experimento sintetico facil: clases separadas por frecuencia. Ese caso llego rapidamente a accuracy perfecta, lo que valido que el pipeline funcionaba, pero tambien mostro que el problema era demasiado sencillo.",
                "Luego se introdujo una dificultad hard: frecuencias mas cercanas, tonos distractores, ruido, desplazamientos temporales y dropout parcial de componentes. Con el banco grueso original el modelo se quedo cerca de 0.53, lo que revelo falta de resolucion en frecuencia.",
                "El banco fine y los filtros aprendibles mejoraron el caso dificil. Esto sugiere que, cuando las clases son ambiguas, los filtros necesitan adaptarse a la estructura del problema.",
                "El gate contextual no mejoro automaticamente. Esta es una conclusion importante: mas dinamismo no siempre produce mejor rendimiento. En este caso, el gate independiente parece actuar como una regularizacion mas estable.",
                "La V2 sparse gate mejoro ligeramente el accuracy y mostro selectividad suave: con umbral 0.7, el modelo concentra gates fuertes en aproximadamente dos bandas, aunque muchas bandas siguen parcialmente abiertas.",
            ],
        )

        rows = []
        for run, label, summary in summaries:
            rows.append(
                [
                    label,
                    str(summary.get("best_epoch", "-")),
                    format_metric(summary.get("best_val_acc", "-")),
                    format_metric(summary.get("final_val_acc", "-")),
                    format_metric(summary.get("final_val_gate_mean", "-")),
                    format_metric(summary.get("final_val_active_bands", "-")),
                ]
            )
        add_table_page(
            pdf,
            "6. Resumen De Experimentos Locales",
            ["Configuracion", "Best Epoch", "Best Val Acc", "Final Val Acc", "Final Gate", "Active Bands"],
            rows,
        )

        snr_rows = read_snr("hard_sparse_gate")
        if snr_rows:
            add_table_page(
                pdf,
                "7. Robustez A Ruido - Sparse Gate",
                ["SNR", "Loss", "Accuracy", "Gate Mean", "Active Bands"],
                [
                    [
                        row.get("snr_db", ""),
                        f"{float(row.get('loss', 0)):.3f}",
                        f"{float(row.get('acc', 0)):.3f}",
                        f"{float(row.get('gate_mean', 0)):.3f}" if row.get("gate_mean") else "-",
                        f"{float(row.get('active_bands', 0)):.2f}" if row.get("active_bands") else "-",
                    ]
                    for row in snr_rows
                ],
            )

        figure_pages = [
            (
                "Training Curves - Sparse Gate L1=0.05",
                ROOT / "outputs" / "hard_sparse_gate_l005" / "figures" / "synthetic_training_curves.png",
                "Loss, accuracy y metricas de selectividad durante entrenamiento.",
            ),
            (
                "Gates Por Clase - Sparse Gate L1=0.05",
                ROOT / "outputs" / "hard_sparse_gate_l005" / "figures" / "synthetic_gates_by_class.png",
                "Promedio de gates por clase y banda. Ayuda a ver que regiones de frecuencia usa cada clase.",
            ),
            (
                "Energia Por Clase - Sparse Gate L1=0.05",
                ROOT / "outputs" / "hard_sparse_gate_l005" / "figures" / "synthetic_energy_by_class.png",
                "Energia logaritmica por banda. Sirve para comparar energia fisica contra gate aprendido.",
            ),
            (
                "SNR Sweep - Sparse Gate",
                ROOT / "outputs" / "hard_sparse_gate" / "figures" / "snr_sweep_hard.png",
                "Accuracy y selectividad bajo diferentes niveles de ruido.",
            ),
        ]
        for title, image_path, note in figure_pages:
            add_image_page(pdf, title, image_path, note)

        add_text_page(
            pdf,
            "8. Comparacion Con Otros Modelos",
            [
                "## MFCC + MLP",
                "Usa caracteristicas manuales compactas. Es rapido y fuerte como baseline clasico, pero pierde informacion fina de waveform y no aprende el frontend.",
                "## Mel-spectrogram + CNN",
                "Convierte audio en una representacion tiempo-frecuencia fija y usa convoluciones 2D. Es una arquitectura fuerte y comun para clasificacion de audio, pero la transformacion mel es predefinida.",
                "## Conv1D sobre waveform",
                "Aprende filtros directamente desde la senal cruda. Es flexible, pero la primera capa puede ser menos interpretable si aprende kernels libres sin restriccion fisica.",
                "## SincNet",
                "SincNet aprende solo las frecuencias de corte baja y alta de filtros pasa banda. Esto ofrece filtros compactos con significado fisico claro y es una referencia directa para una futura V3 parametrizada.",
                "## LEAF / EfficientLEAF",
                "LEAF propone un frontend aprendible de audio que reemplaza mel-filterbanks con filtros, pooling, compresion y normalizacion aprendibles. EfficientLEAF muestra que estos frontends pueden ser mucho mas eficientes, pero tambien advierte que no siempre superan a mel-filterbanks fijos.",
                "## Nuestra EGFN",
                "La EGFN se diferencia por hacer explicita la energia por banda y por usar gates interpretables para modular la salida. No busca ser un reemplazo universal, sino un bloque investigable, liviano y visualizable.",
            ],
        )

        add_text_page(
            pdf,
            "9. Limitaciones Y Siguientes Pasos",
            [
                "Los resultados actuales son sobre datos sinteticos. Eso es correcto para validar la idea, pero no demuestra desempeno en audio real.",
                "El modo hard es intencionalmente ambiguo. La accuracy alrededor de 0.70-0.75 indica que el modelo aprende senal util, pero todavia hay margen para mejorar arquitectura y datos.",
                "La sparse gate mostro selectividad suave, no apagado binario fuerte. Un siguiente paso razonable es agregar gate sharpness o temperatura para empujar gates hacia decisiones mas cercanas a 0/1.",
                "La mejora de mayor valor investigativo seria V3: filtros parametrizados tipo SincNet, aprendiendo frecuencias de corte sin perder interpretabilidad.",
                "Luego se deberia pasar a datasets reales pequenos como Free Spoken Digit Dataset y despues a Speech Commands o ESC-50.",
            ],
        )

        add_text_page(
            pdf,
            "10. Referencias",
            [
                "Ravanelli, M. and Bengio, Y. Speech and Speaker Recognition from Raw Waveform with SincNet. arXiv:1812.05920. https://arxiv.org/abs/1812.05920",
                "Zeghidour, N. et al. LEAF: A Learnable Frontend for Audio Classification. arXiv:2101.08596. https://arxiv.org/abs/2101.08596",
                "Schluter, J. and Gutenbrunner, G. EfficientLEAF: A Faster LEarnable Audio Frontend of Questionable Use. arXiv:2207.05508. https://arxiv.org/abs/2207.05508",
                "Proyecto local: Energy-Gated Frequency Neuron, implementado en PyTorch con datos sinteticos controlados.",
            ],
        )

    print(REPORT_PATH)


if __name__ == "__main__":
    main()
