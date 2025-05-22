document.addEventListener("DOMContentLoaded", function () {
    // Inicializar tooltips de Bootstrap
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));

    // Mostrar/ocultar rango de minmax según método de normalización
    const methodSelect = document.getElementById("normalize_method");
    const minmaxRangeGroup = document.getElementById("minmax-range-group");

    function toggleMinmaxRange() {
        minmaxRangeGroup.style.display = methodSelect.value === "minmax" ? "block" : "none";
    }

    methodSelect.addEventListener("change", toggleMinmaxRange);
    toggleMinmaxRange(); // Estado inicial

    // Capturar y validar formulario antes de enviar
    const form = document.getElementById("procesar-form");

    form.addEventListener("submit", function (e) {
        e.preventDefault(); // Prevenir envío automático
         window.scrollTo({ top: 0, behavior: "smooth" });

        let errors = [];

        // Validar que haya imagen seleccionada
        const imagen = document.getElementById("file").files[0];
        if (!imagen) {
            errors.push("Debes seleccionar una imagen.");
        } else {
            // Opcional: validar tipo y tamaño de archivo
            const allowedTypes = ["image/png", "image/jpeg"];
            if (!allowedTypes.includes(imagen.type)) {
                errors.push("Formato de imagen no válido. Usa PNG o JPEG.");
            }
            // Por ejemplo, máximo 5MB
            const maxSizeMB = 5;
            if (imagen.size > maxSizeMB * 1024 * 1024) {
                errors.push(`El archivo no debe superar ${maxSizeMB} MB.`);
            }
        }

        // Mostrar errores si los hay
        if (errors.length > 0) {
            alert(errors.join("\n"));
            return;
        }

        // Preparar datos para enviar vía fetch (AJAX)
        const formData = new FormData(form);

        // Opcional: mostrar botón como "Procesando..."
        const submitBtn = document.getElementById("submit");
        submitBtn.disabled = true;
        submitBtn.textContent = "Procesando...";

        fetch(form.action, {
            method: "POST",
            body: formData,
        })
            .then(response => {
                if (!response.ok) throw new Error("Error en la respuesta del servidor");
                return response.json(); // Asumiendo que tu backend devuelve JSON
            })
            .then(data => {
                // Mostrar resultado
                const resultadoDiv = document.getElementById("resultado");
                const imagenProcesada = document.getElementById("imagen-procesada");
                const prediccionLabel = document.getElementById("prediccion-label");
                const descriptionLabel = document.getElementById("description-label");

                // Supongamos que data tiene { imagen_url, prediccion }
                imagenProcesada.src = data.imagen_url;
                console.log("Imagen URL:", data.imagen_url);
                // Agrega esta línea para depurar:
                console.log("Valor recibido para data.prediccion:", data.prediccion, typeof data.prediccion);
                const clases = {
                    0: "Glioma Tumor",
                    1: "Meningioma Tumor",
                    2: "No Tumor",
                    3: "Pituitary Tumor"
                };

                const descripciones = {
                    0: "Tumor cerebral que se origina en las células gliales, que dan soporte a las neuronas. Puede ser agresivo y afectar distintas zonas del cerebro.",
                    1: "Tumor generalmente benigno que se forma en las meninges, las membranas que rodean el cerebro y la médula espinal. Suele crecer lentamente.",
                    2: "No se detectaron tumores.",
                    3: "Tumor que se desarrolla en la glándula pituitaria, ubicada en la base del cerebro. Puede afectar la producción hormonal del cuerpo."
                };

                const claseTexto = clases[data.prediccion] || `Clase desconocida (${data.prediccion})`;
                prediccionLabel.textContent = `Predicción: ${claseTexto}`;
                descriptionLabel.textContent = descripciones[data.prediccion] || "Descripción no disponible.";
                resultadoDiv.style.display = "block";
            })
            .catch(error => {
                alert("Error al procesar la imagen: " + error.message);
            })
            .finally(() => {
                submitBtn.disabled = true;
                submitBtn.textContent = "Procesar";
            });
    });

    document.getElementById("reset").addEventListener("click", function () {
    location.reload();
});
});
