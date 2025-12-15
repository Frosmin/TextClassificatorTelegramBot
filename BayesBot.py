import logging
import os
from dotenv import load_dotenv
import pytz
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, JobQueue, CallbackQueryHandler
from telegram.constants import ParseMode 
from trainData import datos_entrenamiento
from DataProcess import procesar_texto_desde_cero
from LazyBayes import NaiveBayesNativo
import json

modelo = NaiveBayesNativo()
load_dotenv()

async def button_handler(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    

    await query.answer()
    
    if query.data == 'pago_qr':
        await query.edit_message_text(
            text="**Opción: Pago por QR**\n\n"
                 "Escanea el código QR adjunto (imaginario por ahora) o usa este ID: `12345678`\n\n"
                 "*Importante:* Envía una captura del comprobante aquí cuando termines.",
            parse_mode=ParseMode.MARKDOWN
        )
        
    elif query.data == 'pago_banco':
        await query.edit_message_text(
            text="**Opción: Transferencia Bancaria**\n\n"
                 "🏦 **Banco:** Banco Nacional\n"
                 "👤 **Titular:** Simon.\n"
                 "🔢 **Cuenta:** 123412312\n"
                 "🆔 **NIT/CI:** 555666777\n\n"
                 "⚠️ *Importante:* Envía una captura del comprobante aquí cuando termines.",
            parse_mode=ParseMode.MARKDOWN
        )
        
    elif query.data == 'cancelar':
        await query.edit_message_text(text="Entendido, hemos cancelado el proceso de compra. Avísanos si necesitas algo más.")




async def accion_compra(update: Update, context: CallbackContext):
    keyboard = [
        [
            InlineKeyboardButton("📲 Pagar por QR", callback_data='pago_qr'),
            InlineKeyboardButton("🏦 Transferencia", callback_data='pago_banco'),
        ],
        [InlineKeyboardButton("❌ Cancelar", callback_data='cancelar')]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    texto = (
        "**¡Excelente decisión!**\n\n"
        "Para finalizar tu pedido, por favor selecciona tu método de pago preferido:"
    )
    
    await update.message.reply_text(texto, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)

async def accion_catalogo(update: Update, context: CallbackContext):
    await update.message.reply_text("Claro, estoy subiendo el catálogo para ti. Un momento por favor...")
    
    nombre_archivo = 'herramientas_proyecto_IA2.pdf'

    try:
        with open(nombre_archivo, 'rb') as documento:
            
            await update.message.reply_document(
                document=documento,
                caption="Aquí tienes nuestra lista de precios y productos actualizada",
                filename="Catalogo.pdf" 
            )
            
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {nombre_archivo}")
        await update.message.reply_text("Lo siento, no encuentro el archivo del catálogo en el sistema. Contacta a un humano.")
    except Exception as e:
        print(f"ERROR AL ENVIAR: {e}")
        await update.message.reply_text("Ocurrió un error al intentar enviarte el archivo.")

async def accion_soporte(update: Update, context: CallbackContext):
    texto = (
        "**Soporte Técnico**\n"
        "Lamento que tengas problemas. Un técnico revisará tu caso.\n"
        "Por favor, envíame una foto del error si es posible."
    )
    await update.message.reply_text(texto, parse_mode=ParseMode.MARKDOWN)


async def accion_ubicacion(update: Update, context: CallbackContext):
    await update.message.reply_text("Nos encontramos aquí:")
    await update.message.reply_location(latitude=-17.3933818, longitude=-66.1460324)


async def accion_generica(update: Update, context: CallbackContext, categoria: str):
    await update.message.reply_text(f"Entendido, tu mensaje es de tipo: *{categoria.upper()}*. En breve te atendemos.", parse_mode=ParseMode.MARKDOWN)


ACCIONES = {
    "compra": accion_compra,
    "catalogo": accion_catalogo,
    "soporte": accion_soporte,
    "consulta": accion_ubicacion,

}

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Hola, Bienvenid@, yo soy BayesBot! Dame tu consulta y te asignaré al personal adecuado para que te ayude.')


async def help_command(update: Update, context: CallbackContext) -> None:
    help_text = (
        "🤖 **BayesBot Commands**\n\n"
        "/start - Start the bot and get a welcome message.\n"
        "/help - Show this help message.\n\n"
        "Just type a sentence, and I'll use a **Bayesian Classifier** "
        "to determine its category or intent."
    )
    
    # Use context.bot.send_message if reply_text isn't suitable, 
    # but reply_text is usually better for direct replies.
    # ParseMode.MARKDOWN_V2 is often used for rich text formatting.
    await update.message.reply_markdown_v2(
        help_text.replace('.', r'\.').replace('-', r'\-')
    )

async def handle_message(update: Update, context: CallbackContext) -> None:
    """Classifies the incoming text message using your Bayesian model."""
    user_text = update.message.text

    # 1. Predecir
    prediccion, scores = modelo.predict(user_text)
    categoria_detectada = prediccion.lower() # Aseguramos minúsculas para buscar en el diccionario

    print(f"Mensaje: {user_text} | Clasificado como: {categoria_detectada}")

    # 2. Buscar la función correspondiente en el diccionario
    funcion_a_ejecutar = ACCIONES.get(categoria_detectada)

    # 3. Ejecutar la acción
    if funcion_a_ejecutar:
        # Si existe una función específica, la ejecutamos
        await funcion_a_ejecutar(update, context)
    else:
        # Si no hay función específica (ej. 'saludo' o 'queja'), usamos la genérica
        await accion_generica(update, context, categoria_detectada)

    

# --- End Handler Functions ---

def main() -> None:
    # Use a timezone provided by pytz (e.g., 'America/New_York')
    # Change 'America/New_York' to your desired timezone if needed.
    
    # Instanciamos la clase
    # Separamos X e y

    datos_entrenamiento_procesados = []

    for dato in datos_entrenamiento:
        datos_entrenamiento_procesados.append((procesar_texto_desde_cero(dato[0]),dato[1]))

    mensajes_train = [d[0] for d in datos_entrenamiento_procesados]
    categorias_train = [d[1] for d in datos_entrenamiento_procesados]

    # Entrenamos
    modelo.fit(mensajes_train, categorias_train)
    
    print("💾 Exportando modelo para n8n...")
    
    datos_para_n8n = {
        "vocabulario": list(modelo.vocabulario),
        "priors": modelo.log_priors,          # Probabilidad base de cada categoría
        "likelihoods": modelo.log_likelihoods # Probabilidad de cada palabra por categoría
    }

    nombre_json = 'modelo_entrenado.json'
    
    with open(nombre_json, 'w', encoding='utf-8') as f:
        json.dump(datos_para_n8n, f, indent=4, ensure_ascii=False)
        
    print(f"✅ ¡Listo! El archivo '{nombre_json}' ha sido creado.")
    print("Copia el contenido de ese archivo y úsalo en tu flujo de n8n.")
    
    timezone = pytz.timezone('America/La_Paz')


    # 1. Initialize the Application with an explicit timezone setting
    application = (
        Application.builder()
        .token(os.environ.get("TOKEN")) # <-- Pass the configured job_queue instance here
        .build()
    )

    # ... The rest of your main function (add_handler, run_polling) remains the same ...
    
    # 2. Add Handlers to the Application (as before)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
# 2. Add Handlers to the Application (as before)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    
    # AGREGAR ESTA LÍNEA: Manejador para los botones
    application.add_handler(CallbackQueryHandler(button_handler))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    


    # 3. Start the Bot (Polling)
    print("El bot esta listo....")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()