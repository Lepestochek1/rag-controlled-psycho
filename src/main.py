import os
import asyncio
import logging
from dotenv import load_dotenv

from services import EvaluationService, TopicValidationService, InputControlService, PostProcessingService
from rag import extract_preview_text
from database import db

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# Налаштування
# =========================
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
import os
key = os.getenv("OPENAI_API_KEY")
if key:
    print(f"DEBUG: Key starts with {key[:10]} and ends with {key[-4:]}")
else:
    print("DEBUG: Key NOT FOUND")
    print(os.getenv("MY_TEST_KEY"))
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY не знайдено. Перевір .env файл")

print("✅ OPENAI_API_KEY завантажено")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MY_TEST = os.getenv("MY_TEST_KEY")

print(f"DEBUG: Тепер ключ починається з: {OPENAI_API_KEY[:12]}")
print(f"DEBUG: Тестовий ключ: {MY_TEST}")
logging.basicConfig(level=logging.INFO)

DATA_DIR = "data"
BOOKS_DIR = r"F:\Навчання\4 курс 2 семестр\Diploma\books"
DB_FOLDER = "data/chroma_db"

ADMIN_IDS = {6395016659}

os.makedirs(BOOKS_DIR, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)


# =========================
# RAG-сервіс
# =========================
class RAGService:
    def __init__(self, books_dir: str, db_dir: str, api_key: str):
        self.books_dir = books_dir
        self.db_dir = db_dir

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )

        self._load_or_create_db()

    def _load_or_create_db(self):
        if not os.listdir(self.db_dir):
            self.rebuild_from_folder()
        else:
            self.db = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )
        self._build_chain()

    def rebuild_from_folder(self):
        logging.info("🔄 Індексація PDF-файлів...")

        docs = []
        for file in os.listdir(self.books_dir):
            if file.endswith(".pdf"):
                docs.extend(
                    PyPDFLoader(os.path.join(self.books_dir, file)).load()
                )

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        ).split_documents(docs)

        self.db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )

        self._build_chain()

    def _build_chain(self):
        retriever = self.db.as_retriever(search_kwargs={"k": 5})

        prompt = PromptTemplate.from_template("""
Ти — фахівець із психосоматики, який надає структуровані та обґрунтовані пояснення взаємозв’язку між психоемоційним станом людини та фізичними симптомами.

На основі наданого КОНТЕКСТУ сформуй відповідь на ПИТАННЯ, дотримуючись такої структури:
1. Фізичне блокування — опиши можливі тілесні прояви або симптоми.
2. Емоційне блокування — визнач емоції або переживання, що можуть бути пов’язані з проблемою.
3. Ментальне блокування — проаналізуй переконання, установки або мисленнєві патерни, що впливають на стан.
4. Рекомендації — надай узагальнені поради щодо покращення стану (без медичних діагнозів або лікування).

Вимоги до відповіді:
- Використовуй лише інформацію з КОНТЕКСТУ.
- Відповідь має бути логічною, чіткою та структурованою.
- Уникай категоричних медичних висновків і діагнозів.
- Формулюй пояснення у нейтральному, науково-інформаційному стилі.

Якщо ПИТАННЯ не стосується психосоматики або КОНТЕКСТ не містить релевантної інформації — повідом, що ти спеціалізуєшся виключно на психосоматиці, і не можеш надати відповідь.

КОНТЕКСТ:
{context}

ПИТАННЯ:
{question}
""")

        self.chain = (
            {
                "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                "question": RunnablePassthrough()
            }
            | prompt
            | ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                openai_api_key=OPENAI_API_KEY
            )
            | StrOutputParser()
        )

    def ask(self, query: str) -> str:
        return self.chain.invoke(query)


# =========================
# Telegram-бот
# =========================
class PsychoBot:
    def __init__(self, token: str, rag: RAGService, evaluator: EvaluationService):
        self.bot = Bot(token=token)
        self.dp = Dispatcher()

        self.rag = rag
        self.evaluator = evaluator
        self.validator = TopicValidationService()
        self.input_control = InputControlService()
        self.postprocessor = PostProcessingService()
        # Команди
        self.dp.message(Command("start"))(self.cmd_start)
        self.dp.message(Command("help"))(self.cmd_help)
        self.dp.message(Command("about"))(self.cmd_about)
        self.dp.message(Command("upload"))(self.cmd_upload)

        # 🔥 КНОПКА (ВАЖЛИВО — ДО handle_text)
        self.dp.message(F.text == "📋 Список команд")(self.show_commands)

        # Інше
        self.dp.message(F.document)(self.handle_document)
        self.dp.message(F.text)(self.handle_text)

    def is_admin(self, user_id: int) -> bool:
        return user_id in ADMIN_IDS

    def keyboard(self):
        return ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text="📋 Список команд")]],
            resize_keyboard=True
        )

    # =========================
    # ЄДИНЕ ДЖЕРЕЛО КОМАНД
    # =========================
    def get_commands_text(self, user_id: int):
        text = (
            "📋 Доступні команди:\n\n"
            "/start — запуск бота\n"
            "/help — список команд\n"
            "/about — опис можливостей\n"
        )

        if self.is_admin(user_id):
            text += "/upload — завантажити PDF\n"

        return text

    # =========================
    # HANDLERS
    # =========================
    async def cmd_start(self, message: Message):
        await message.answer(
            "🌿 Напишіть симптом або орган.",
            reply_markup=self.keyboard()
        )

    async def cmd_help(self, message: Message):
        await message.answer(self.get_commands_text(message.from_user.id))

    async def show_commands(self, message: Message):
        await message.answer(self.get_commands_text(message.from_user.id))

    async def cmd_about(self, message: Message):
        await message.answer("Бот для психосоматичного аналізу (RAG + PDF).")

    async def cmd_upload(self, message: Message):
        if not self.is_admin(message.from_user.id):
            await message.answer("⛔ Недостатньо прав.")
            return
        await message.answer("📄 Надішліть PDF-файл.")

    async def handle_document(self, message: Message):
        if not self.is_admin(message.from_user.id):
            return

        if not message.document.file_name.endswith(".pdf"):
            await message.answer("❌ Потрібен PDF-файл.")
            return

        file = await self.bot.get_file(message.document.file_id)
        path = os.path.join(BOOKS_DIR, message.document.file_name)
        await self.bot.download_file(file.file_path, path)

        await message.answer("🔍 Перевіряю тематику книги...")

        preview = extract_preview_text(path)

        loop = asyncio.get_event_loop()
        is_valid = await loop.run_in_executor(None, self.validator.validate, preview)

        db.save_book(message.document.file_name, is_valid)

        if not is_valid:
            os.remove(path)
            await message.answer("⛔ Книга не відповідає тематиці психосоматики.")
            return

        await message.answer("✅ Тематика підтверджена. Оновлюю базу...")
        self.rag.rebuild_from_folder()
        await message.answer("📚 База знань оновлена.")

    async def handle_text(self, message: Message):
        text = message.text

        # =========================
        # 🔴 1. Intent detection
        # =========================
        intent = self.input_control.detect_intent(text)

        if intent == "empty":
            await message.answer("Будь ласка, введіть запит.")
            return

        if intent == "joke":
            await message.answer("🙂 Я спеціалізуюся на психосоматиці. Спробуйте поставити запитання по темі.")
            return

        # =========================
        # 🔴 2. Risk detection
        # =========================
        if self.input_control.detect_risk(text):
            await message.answer(
                "⚠️ Ваш запит може свідчити про емоційно складний стан.\n\n"
                "Я можу надати лише загальну інформацію. "
                "Будь ласка, зверніться до психолога або лікаря 💙"
            )
            return

        # =========================
        # 🔴 3. Захист від кнопки
        # =========================
        if text == "📋 Список команд":
            return

        await self.bot.send_chat_action(message.chat.id, "typing")

        loop = asyncio.get_event_loop()

        # 🔹 RAG відповідь
        response = await loop.run_in_executor(None, self.rag.ask, text)
        # 🔴 ПОСТ-ПРОЦЕСИНГ
        response = self.postprocessor.process(response)
        # 🔹 Оцінка
        evaluation = await loop.run_in_executor(
            None, self.evaluator.evaluate, text, response
        )

        await message.answer(
            f"{response}\n\n---\n📊 {evaluation}",
            reply_markup=self.keyboard()
        )

    async def run(self):
        logging.info("🤖 Бот запущено")
        await self.dp.start_polling(self.bot)
# =========================
# Запуск
# =========================
if __name__ == "__main__":
    rag = RAGService(BOOKS_DIR, DB_FOLDER, OPENAI_API_KEY)
    evaluator = EvaluationService()
    bot = PsychoBot(BOT_TOKEN, rag, evaluator)
    asyncio.run(bot.run())
